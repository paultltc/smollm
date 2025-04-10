import copy
import os

from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING, AutoConfig

from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig
from m4.models.custom_modules import FreezeConfig

logger = logging.get_logger(__name__)

class VBertTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        embed_dim (`int`, *optional*, defaults to 1152):
            Dimensionality of the encoder layers and the pooler layer. (elsewhere referred to as `embed_dim`)
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
    """
    model_type = "vbert"

    def __init__(
        self,
        # Case for when vllama3 is from the hub with no vision_model_name
        text_model_name="EuroBERT/EuroBERT-210m",
        **kwargs,
    ):
        self.text_model_name = text_model_name
        text_config = AutoConfig.from_pretrained(text_model_name, trust_remote_code=True)

        if hasattr(text_config, "text_config"):
            text_config = text_config.text_config

        # Hidden size is the same 
        if hasattr(text_config, "hidden_size"):
            self.hidden_size = text_config.hidden_size
        else:
            raise ValueError("text_config must have a hidden_size")
        
        if hasattr(text_config, "num_hidden_layers"):
            self.num_hidden_layers = text_config.num_hidden_layers
        else:
            raise ValueError("text_config must have a num_hidden_layers")
        
        if hasattr(text_config, "mlp_bias"):
            self.mlp_bias = text_config.mlp_bias
        else:
            self.mlp_bias = False

        super().__init__(**kwargs)

class VBertVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        embed_dim (`int`, *optional*, defaults to 1152):
            Dimensionality of the encoder layers and the pooler layer. (elsewhere referred to as `embed_dim`)
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
    """
    model_type = "vbert"
    attribute_map = {
        "hidden_size": "embed_dim",
    }

    def __init__(
        self,
        # Case for when vllama3 is from the hub with no vision_model_name
        vision_model_name="HuggingFaceM4/siglip-so400m-14-384",
        **kwargs,
    ):
        self.vision_model_name = vision_model_name
        vision_config = AutoConfig.from_pretrained(vision_model_name, trust_remote_code=True)
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config

        # vllama3 case (necessary for loading the vllama3 model)
        if hasattr(vision_config, "embed_dim"):
            self.embed_dim = vision_config.embed_dim
        # clip case (necessary for initialization)
        elif hasattr(vision_config, "hidden_size"):
            self.embed_dim = vision_config.hidden_size
        else:
            raise ValueError("vision_config must have a hidden_size or embed_dim")

        if hasattr(vision_config, "image_size"):
            self.image_size = vision_config.image_size
        else:
            raise ValueError("vision_config must have an image_size")

        if hasattr(vision_config, "patch_size"):
            self.patch_size = vision_config.patch_size
        else:
            raise ValueError("vision_config must have a patch_size")

        if hasattr(vision_config, "num_hidden_layers"):
            self.num_hidden_layers = vision_config.num_hidden_layers
        else:
            raise ValueError("vision_config must have a num_hidden_layers")

        if hasattr(vision_config, "intermediate_size"):
            self.intermediate_size = vision_config.intermediate_size
        else:
            raise ValueError("vision_config must have an intermediate_size")

        super().__init__(**kwargs)

class VBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SmolVLMModel`]. It is used to instantiate a
    SmolVLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the model of the SmolVLM
    [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cache the key/value pairs of the attention mechanism. Only
            relevant if `config.is_decoder=True`.
        image_token_id (`int`, *optional*, defaults to 128257):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`IdeficsVisionConfig` or `dict`, *optional*, defaults to `IdeficsVisionConfig`):
            Custom vision config or dict for the vision tower
        text_config (`PretrainedConfig` or `dict`, *optional*, defaults to `LlamaConfig`):
            Custom text config or dict for the text model
        scale_factor (`int`, *optional*, defaults to 2):
            The scale factor for the image encoder.
        pad_token_id (`int`, *optional*, defaults to 128002):
            The id of the padding token.

    Example:
    ```python
    >>> from transformers import SmolVLMModel, SmolVLMConfig
    >>> # Initializing configuration
    >>> configuration = SmolVLMConfig()
    >>> # Initializing a model from the configuration
    >>> model = SmolVLMModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vbert"
    is_composition = True
    sub_configs = {"text_config": VBertTextConfig, "vision_config": VBertVisionConfig}

    def __init__(
        self,
        use_cache=True,
        tie_word_embeddings=False,
        vision_config=None,
        text_config=None,
        freeze_config=None,
        image_token_id=128257,
        pad_token_id=128_002,
        pixel_shuffle_factor=4,
        use_resampler=False,
        additional_vocab_size=0,
        neftune_noise_alpha=0.0,
        _attn_implementation="sdpa",
        **kwargs,
    ):        
        self.image_token_id = image_token_id
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.scale_factor = pixel_shuffle_factor
        self.additional_vocab_size = additional_vocab_size
        self._attn_implementation = _attn_implementation

        # Vision config
        if vision_config is None:
            self.vision_config = VBertVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = VBertVisionConfig(**vision_config)
        elif isinstance(vision_config, VBertVisionConfig):
            self.vision_config = vision_config

        # Text config
        if text_config is None:
            self.text_config = VBertTextConfig()
        elif isinstance(text_config, dict):
            self.text_config = VBertTextConfig(**text_config)
        elif isinstance(text_config, VBertTextConfig):
            self.text_config = text_config

        # Freezing layers
        # if freeze_config is None:
        #     self.freeze_config = FreezeConfig()
        # elif isinstance(freeze_config, dict):
        #     self.freeze_config = FreezeConfig(**freeze_config)
        # elif isinstance(freeze_config, FreezeConfig):
        #     self.freeze_config = freeze_config
        self.freeze_config = freeze_config

        # Pixel shuffle factor
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.use_resampler = use_resampler

        self.neftune_noise_alpha = neftune_noise_alpha

        pad_token_id = self.text_config.pad_token_id if hasattr(self.text_config, "pad_token_id") and self.text_config.pad_token_id is not None else pad_token_id

        super().__init__(**kwargs, pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        output["model_type"] = self.__class__.model_type
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        # output["freeze_config"] = self.freeze_config.to_dict()

        return output
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        outputs = super(VBertConfig, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        return outputs
