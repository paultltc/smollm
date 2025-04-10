from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.processing_utils import ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin, pad_without_fast_tokenizer_warning, _torch_collate_batch

from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np

import logging
logger = logging.getLogger(__name__)

# def process_batch(self, batch: List[Union[List[int], Any, Dict[str, Any]]], **processor_kwargs) -> Dict[str, Any]:
#     """
#     Process a batch of examples.
#     """
#     query = " ".join([self.tokenizer.decode(ex) for ex in batch])

#     texts = []
#     images = []
#     for ex in batch:
#         message = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": query},
#                     {"type": "image"}
#                 ]
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {"type": "text", "text": ex["text"]},
#                 ]
#             }
#         ]
#         prompt = self.processor.apply_chat_template(message, add_generation_prompt=False).strip()
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         texts.append(prompt)
#         images.append(image)

#     assert len(texts) == len(images), f"Number of samples in texts and images should be the same. Got {len(texts)} texts and {len(images)} batch of image."

#     return self.processor(text=texts, images=images, **processor_kwargs)

@dataclass
class SimpleDataCollatorForVisionLanguage(DataCollatorMixin):
    processor: ProcessorMixin
    tokenizer: PreTrainedTokenizerBase = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.tokenizer is None and hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            raise ValueError(
                "You need to specify a tokenizer to use `DataCollatorForVisionLanguage` collators."
            )

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("<image>")
        ]

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # if isinstance(examples[0], Mapping):
        #     batch = self.processor(text=[ex["text"] for ex in examples], images=[ex["images"] for ex in examples], return_tensors=self.return_tensors)
        # else:
        #     batch = self.processor(text=examples["text"], images=examples["images"],return_tensors=self.return_tensors)
        batch = self.processor(examples)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        # If special token mask has been preprocessed, pop it from the dict.
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        return batch
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None, **kwargs) -> Tuple[Any, Any]:
        # This is a dummy method that should be implemented in the subclasses
        return inputs, inputs.clone()


@dataclass
class DataCollatorForVisionLanguageModeling(SimpleDataCollatorForVisionLanguage):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        mask_replace_prob (`float`, *optional*, defaults to 0.8):
            The probability with which masked tokens are replaced by the tokenizer's mask token (e.g., `[MASK]`).
            Defaults to 0.8, meaning 80% of the masked tokens will be replaced with `[MASK]`.
            Only works when `mlm` is set to `True`.
        random_replace_prob (`float`, *optional*, defaults to 0.1):
            The probability with which masked tokens are replaced by random tokens from the tokenizer's vocabulary.
            Defaults to 0.1, meaning 10% of the masked tokens will be replaced with random tokens. The remaining
            masked tokens (1 - mask_replace_prob - random_replace_prob) are left unchanged.
            Only works when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    <Example Options and Expectations>

    1. Default Behavior:
        - `mask_replace_prob=0.8`, `random_replace_prob=0.1`.
        - Expect 80% of masked tokens replaced with `[MASK]`, 10% replaced with random tokens, and 10% left unchanged.

    2. All masked tokens replaced by `[MASK]`:
        - `mask_replace_prob=1.0`, `random_replace_prob=0.0`.
        - Expect all masked tokens to be replaced with `[MASK]`. No tokens are left unchanged or replaced with random tokens.

    3. No `[MASK]` replacement, only random tokens:
        - `mask_replace_prob=0.0`, `random_replace_prob=1.0`.
        - Expect all masked tokens to be replaced with random tokens. No `[MASK]` replacements or unchanged tokens.

    4. Balanced replacement:
        - `mask_replace_prob=0.5`, `random_replace_prob=0.4`.
        - Expect 50% of masked tokens replaced with `[MASK]`, 40% replaced with random tokens, and 10% left unchanged.

    Note:
        The sum of `mask_replace_prob` and `random_replace_prob` must not exceed 1. If their sum is less than 1, the
        remaining proportion will consist of masked tokens left unchanged.

    </Tip>
    """

    mlm: bool = True
    mlm_probability: float = 0.5
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        super().__post_init__()
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        if self.mlm_probability < 0 or self.mlm_probability > 1:
            raise ValueError("mlm_probability should be between 0 and 1.")
        if self.mask_replace_prob + self.random_replace_prob > 1:
            raise ValueError("The sum of mask_replace_prob and random_replace_prob should not exceed 1")
        if self.mask_replace_prob < 0 or self.mask_replace_prob > 1:
            raise ValueError("mask_replace_prob should be between 0 and 1.")
        if self.random_replace_prob < 0 or self.random_replace_prob > 1:
            raise ValueError("random_replace_prob should be between 0 and 1.")
    
    def torch_mask_tokens(
            self, 
            inputs: Any, 
            special_tokens_mask: Optional[Any] = None,
            mask_images: bool = False,
        ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Set the image tokens to 0.0 probability if we don't want to mask them
        if not mask_images:
            images_mask = labels == self.image_token_id
            probability_matrix.masked_fill_(images_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1 - self.mask_replace_prob
        # scaling the random_replace_prob to the remaining probability for example if
        # mask_replace_prob = 0.8 and random_replace_prob = 0.1,
        # then random_replace_prob_scaled = 0.1 / 0.2 = 0.5
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob

        # random_replace_prob% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, random_replace_prob_scaled)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(self.tokenizer.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time ((1-random_replace_prob-mask_replace_prob)% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
@dataclass
class DataCollatorForVisionLanguageSFT(SimpleDataCollatorForVisionLanguage):
    """
    Data collator used for supervised finetuning. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        mask_replace_prob (`float`, *optional*, defaults to 0.8):
            The probability with which masked tokens are replaced by the tokenizer's mask token (e.g., `[MASK]`).
            Defaults to 0.8, meaning 80% of the masked tokens will be replaced with `[MASK]`.
            Only works when `mlm` is set to `True`.
        random_replace_prob (`float`, *optional*, defaults to 0.1):
            The probability with which masked tokens are replaced by random tokens from the tokenizer's vocabulary.
            Defaults to 0.1, meaning 10% of the masked tokens will be replaced with random tokens. The remaining
            masked tokens (1 - mask_replace_prob - random_replace_prob) are left unchanged.
            Only works when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    <Example Options and Expectations>

    1. Default Behavior:
        - `mask_replace_prob=0.8`, `random_replace_prob=0.1`.
        - Expect 80% of masked tokens replaced with `[MASK]`, 10% replaced with random tokens, and 10% left unchanged.

    2. All masked tokens replaced by `[MASK]`:
        - `mask_replace_prob=1.0`, `random_replace_prob=0.0`.
        - Expect all masked tokens to be replaced with `[MASK]`. No tokens are left unchanged or replaced with random tokens.

    3. No `[MASK]` replacement, only random tokens:
        - `mask_replace_prob=0.0`, `random_replace_prob=1.0`.
        - Expect all masked tokens to be replaced with random tokens. No `[MASK]` replacements or unchanged tokens.

    4. Balanced replacement:
        - `mask_replace_prob=0.5`, `random_replace_prob=0.4`.
        - Expect 50% of masked tokens replaced with `[MASK]`, 40% replaced with random tokens, and 10% left unchanged.

    Note:
        The sum of `mask_replace_prob` and `random_replace_prob` must not exceed 1. If their sum is less than 1, the
        remaining proportion will consist of masked tokens left unchanged.

    </Tip>
    """

    end_of_utterance_token_id: int = None
    bos_token_id: int = None
    eos_token_id: int = None
    assistant_token_ids: List[int] = None
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        super().__post_init__()
        if self.end_of_utterance_token_id is None:
            raise ValueError(
                "That logic has only been implemented at this point for computing the loss on the assistant answers in"
                " a user/assistant dialogue. We need `end_of_utterance_token_id`."
            )
        if self.bos_token_id is None or self.eos_token_id is None:
            raise ValueError(
                "Case where we don't separate packed sequence by `<BOS>` and `<EOS>` is not supported yet."
            )
        if self.assistant_token_ids is None:
            raise ValueError(
                "We were hoping to mask the part `\nAssistant:` too from the loss computation but"
                " `assistant_token_ids` is not specified."
            )
    
    def torch_mask_tokens(
            self, 
            inputs: Any, 
            special_tokens_mask: Optional[Any] = None
        ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        output_labels = []
        for input_ids_ in inputs:
            labels_ = input_ids_.clone()
            if (labels_ == self.end_of_utterance_token_id).sum() % 2 != 0:
                logger.error(
                    "Did not find an even number of `END_OF_UTTERANCE` tokens in the user/assistant dialogue. Not"
                    " masking the labels."
                )
                output_labels.append(labels_)
                continue

            starts_ends = self.find_delimiters_tokens_to_mask(labels_.tolist())
            for start_index, end_index in starts_ends:
                labels_[start_index:end_index] = -100  # We ignore the user part in a dialogue for the loss computation

            output_labels.append(labels_)

        labels = torch.stack(output_labels)

        return inputs, labels

    def find_delimiters_tokens_to_mask(self, label_list):
        starts_ends_list = []
        start, end = None, None
        counter_eou = 0

        for idx, l_ in enumerate(label_list):
            if l_ == self.bos_token_id:
                assert start is None and end is None, (idx, start, end)
                start = idx
            elif l_ == self.end_of_utterance_token_id:
                counter_eou += 1
                if counter_eou % 2 != 0:
                    assert start is not None and end is None, (idx, start, end)
                    assert label_list[idx + 1 : idx + 1 + len(self.assistant_token_ids)] == self.assistant_token_ids
                    end = idx + 1 + len(self.assistant_token_ids)
                    starts_ends_list.append((start, end))
                    start, end = None, None
                else:
                    assert start is None and end is None, (idx, start, end)
                    if idx + 1 < len(label_list) and label_list[idx + 1] != self.eos_token_id:
                        start = idx + 1
            elif l_ == self.eos_token_id:
                assert start is None and end is None, (idx, start, end)
        assert start is None and end is None, (idx, start, end)

        return starts_ends_list