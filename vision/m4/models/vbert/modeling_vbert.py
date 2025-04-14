import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List

# from transformers.models.smolvlm import SmolVLMModel, SmolVLMPreTrainedModel
from transformers.cache_utils import DynamicCache

from m4.models import DecoupledEmbedding
from m4.models.custom_modules import VLOOMPreTrainedModelBase, FreezeConfig
from m4.training.utils import (
    compute_linear_tflops_per_batch_per_gpu,
    compute_perceiver_tflops_per_batch_per_gpu,
    compute_tflops_per_batch_per_gpu,
    deepspeed_gathered_parameters_context_manager,
    freeze_model,
    regex_lookup
)
from m4.training.setup_vision_model import vision_model_name2model
from m4.training.setup_text_model import language_model_name2model

from .configuration_vbert import VBertConfig

from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bert.modeling_bert import BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput
from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint

from dataclasses import dataclass

from transformers import logging

logger = logging.get_logger(__name__)

@dataclass
class VBertBaseModelOutput(BaseModelOutput):
    """
    Base class for SmolVLM model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class VBertMaskedLMOutput(MaskedLMOutput):
    """
    Base class for SmolVLM model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor`): 
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None

class VBertSimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)

class VBertConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.pixel_shuffle_factor
        self.modality_projection = VBertSimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states

class VBertPreTrainedModel(VLOOMPreTrainedModelBase):
    config_class = VBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VBertDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def override_vision_model_wrapper(cls, model, config, vision_model_name, vision_model_params, torch_dtype):
        # this can be called via from_pretrained from a class w/ head or w/o head so we extract the beheaded model version
        beheaded_model = model.model if hasattr(model, "model") else model
        cls.override_vision_model(beheaded_model, vision_model_name, vision_model_params, torch_dtype)
        beheaded_model.freeze_relevant_params(config)

class VBertModel(VBertPreTrainedModel):
    """
    A subclass of Idefics3Model. We do *not* remove or block the call to inputs_merger
    in forward. Instead, we override inputs_merger here with custom logic.
    """

    def __init__(self, config: VBertConfig):
        super().__init__(config)

        self.vision_model = VBertModel.init_vision_model(config)
        self.connector = VBertConnector(config)
        self.text_model = VBertModel.init_language_model(config)

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )
        self.image_token_id = self.config.image_token_id

        self.post_init()

        self.freeze_relevant_params(config)

    @staticmethod
    def init_vision_model(config: VBertConfig):
        vision_model_name = config.vision_config.vision_model_name

        # Create an uninitialized vision_model to insert into the main model.
        vision_model_config = AutoConfig.from_pretrained(vision_model_name, trust_remote_code=True)

        # Override image_size if we want to increase it compared to pretraining
        if hasattr(vision_model_config, "vision_config"):
            vision_model_config.vision_config.image_size = config.vision_config.image_size
        else:
            vision_model_config.image_size = config.vision_config.image_size

        vision_model = AutoModel.from_config(vision_model_config, trust_remote_code=True)
        extractor = regex_lookup(vision_model_name, vision_model_name2model)

        return extractor(vision_model)

    @staticmethod
    def init_language_model(config: VBertConfig):
        language_model_name = config.text_config.text_model_name

        language_model_config = AutoConfig.from_pretrained(language_model_name, trust_remote_code=True)
        
        # if config.text_config._attn_implementation == "flash_attention_2":
        language_model_config._attn_implementation = config._attn_implementation
        torch_dtype = torch.bfloat16 if config._attn_implementation == "flash_attention_2" else None

        model = AutoModel.from_config(language_model_config, trust_remote_code=True, torch_dtype=torch_dtype)
        # extractor = regex_lookup(language_model_name, language_model_name2model)

        embed_layer = DecoupledEmbedding(
            num_embeddings=config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=config.freeze_config["freeze_text_layers"],
            padding_idx=config.pad_token_id,
        )

        model.set_input_embeddings(embed_layer)

        return model

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.

        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032

        Override to set output.requires_grad = True for both the decoder's and vision model's embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = get_lowest_module(self.vision_model).register_forward_hook(
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def freeze_relevant_params(self, config=None):
        config = config or self.config
        freeze_config = FreezeConfig.from_dict(config.freeze_config)

        if freeze_config.freeze_text_layers:
            freeze_model(self.text_model, module_exceptions=freeze_config.freeze_text_module_exceptions)

        if freeze_config.freeze_vision_layers:
            freeze_model(self.vision_model, module_exceptions=freeze_config.freeze_vision_module_exceptions)

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def inputs_merger(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
    ):
        """
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder and that hidden state, after a pixel shuffle operation, is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """
        _, patch_size, _ = image_hidden_states.shape

        image_mask = input_ids == self.image_token_id
        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("At least one sample has <image> tokens not divisible by patch_size.")

        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]

        merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
        return merged_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

            if not any(real_images_inds):
                # no images, leave one empty image.
                real_images_inds[0] = True

            pixel_values = pixel_values[real_images_inds].contiguous()

            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=[pixel_values.shape[i] for i in (0, 2, 3)],
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            # patch_size = self.config.vision_config.patch_size
            # patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            # patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            # patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                # patch_attention_mask=patch_attention_mask,
            ).last_hidden_state

            print(image_hidden_states)

            # Modality projection & resampling
            image_hidden_states = self.connector(image_hidden_states)

            print(image_hidden_states)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if inputs_embeds is not None and image_hidden_states is not None:
            # When we embed, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return VBertBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )

class VBertForMaskedLM(VBertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.image_token_id = self.config.image_token_id
        self.in_features = config.hidden_size
        self.out_features = config.vocab_size
        self.out_additional_features = config.additional_vocab_size

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.model = VBertModel(config)
        self.lm_head = VBertForMaskedLM.init_lm_head(config)
        if self.out_additional_features > 0:
            self.additional_fc = nn.Linear(
                in_features=self.in_features,
                out_features=self.out_additional_features,
                bias=False,
            )

        # Initialize weights and apply final processing
        self.post_init()

        # Freeze relevant parameters
        self.freeze_relevant_params(config)

    @staticmethod
    def init_lm_head(config):
        # TODO: Check if it works with modernbert
        # language_model_name = config.text_config.text_model_name
        # lm_head = AutoModelForMaskedLM.from_pretrained(language_model_name, trust_remote_code=True).lm_head
        return nn.Linear(config.hidden_size, config.vocab_size, False)

    def freeze_relevant_params(self, config=None):
        config = config or self.config
        freeze_config = FreezeConfig.from_dict(config.freeze_config)

        if freeze_config.freeze_lm_head:
            freeze_model(self.lm_head)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_attention_mask: Optional[torch.BoolTensor] = None,
            image_hidden_states: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, VBertMaskedLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `Idefics3ForConditionalGeneration`).
                Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
                computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        # Pass the inputs to VBertModel
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Pass the outputs to the MLM head
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        if self.out_additional_features > 0:
            additional_features = self.additional_fc(hidden_states)
            logits = torch.cat((logits, additional_features), -1)
        logits = logits.float()

        masked_lm_loss = None
        if labels is not None:
            # print the ratio of not ignored tokens
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size + self.out_additional_features), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return VBertMaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def get_model_tflops_per_batch_per_gpu(self, hparams, data_param, tokenizer, max_num_images, max_num_tokens=None):
        config_vl_model = self.config

        language_embed_size = config_vl_model.hidden_size
        num_language_layers = config_vl_model.num_hidden_layers
        ffn_inner_size = config_vl_model.intermediate_size

        vision_config = config_vl_model.vision_config

        # Get vision model blocks infos
        vision_patch_size = vision_config.patch_size
        vision_hidden_size = vision_config.embed_dim
        num_vision_layers = vision_config.num_hidden_layers
        # The +1 is for the CLS token
        single_image_vision_encoder_seq_len = int(((vision_config.image_size // vision_patch_size) ** 2) // (self.config.pixel_shuffle_factor**2))
        vision_exp_factor = vision_config.intermediate_size // vision_hidden_size

        # Get language blocks infos
        language_seq_len = max_num_tokens if max_num_tokens is not None else data_param.max_seq_len
        language_exp_factor = (ffn_inner_size // language_embed_size) if ffn_inner_size is not None else 4

        # Get modality projection infos
        vision_pipeline_output_seq_len = (
            self.config.perceiver_config.resampler_n_latents
            if self.config.use_resampler
            else single_image_vision_encoder_seq_len
        )

        language_tflops_per_batch_per_gpu = compute_tflops_per_batch_per_gpu(
            num_layers=num_language_layers,
            batch_size=hparams.batch_size_per_gpu,
            q_seq_len=language_seq_len,
            k_seq_len=language_seq_len,
            hidden_size=language_embed_size,
            kv_in_dim=language_embed_size,
            ff_exp_factor=language_exp_factor,
            grad_acc_size=hparams.grad_acc_size,
            swiglu=True,
            vocab_size=tokenizer.vocab_size,
            count_backward=True,  # Always True regardless of freezing, because gradients are computed for vision adaptor
            use_grad_checkpointing=hparams.gradient_checkpointing,
        )
        modality_projection_tflops_per_batch_per_gpu = compute_linear_tflops_per_batch_per_gpu(
            batch_size=hparams.batch_size_per_gpu * max_num_images,
            seq_len=vision_pipeline_output_seq_len,
            in_features=vision_hidden_size,
            out_features=language_embed_size,
            count_backward=True,
            use_grad_checkpointing=hparams.gradient_checkpointing,
        )

        vision_tflops_per_batch_per_gpu = compute_tflops_per_batch_per_gpu(
            num_layers=num_vision_layers,
            batch_size=hparams.batch_size_per_gpu * max_num_images,
            q_seq_len=single_image_vision_encoder_seq_len,
            k_seq_len=single_image_vision_encoder_seq_len,
            hidden_size=vision_hidden_size,
            kv_in_dim=vision_hidden_size,
            ff_exp_factor=vision_exp_factor,
            grad_acc_size=hparams.grad_acc_size,
            swiglu=False,
            vocab_size=None,
            count_backward=not hparams.model_config["freeze_config"]["freeze_vision_layers"],
            use_grad_checkpointing=hparams.gradient_checkpointing,
        )
        if self.config.use_resampler:
            perceiver_tflops_per_batch_per_gpu = compute_perceiver_tflops_per_batch_per_gpu(
                num_layers=self.config.perceiver_config.resampler_depth,
                batch_size=hparams.batch_size_per_gpu * max_num_images,
                q_seq_len=self.config.perceiver_config.resampler_n_latents,
                vision_embed_seq_len=single_image_vision_encoder_seq_len,
                q_k_v_input_dim=vision_hidden_size,
                attention_hidden_size=self.config.perceiver_config.resampler_n_heads
                * self.config.perceiver_config.resampler_head_dim,
                ff_exp_factor=4,
                count_backward=True,
                use_grad_checkpointing=hparams.gradient_checkpointing,
            )
            tflop_count = (
                language_tflops_per_batch_per_gpu
                + modality_projection_tflops_per_batch_per_gpu
                + perceiver_tflops_per_batch_per_gpu
                + vision_tflops_per_batch_per_gpu
            )
        else:
            tflop_count = (
                language_tflops_per_batch_per_gpu
                + modality_projection_tflops_per_batch_per_gpu
                + vision_tflops_per_batch_per_gpu
            )
        return tflop_count
    