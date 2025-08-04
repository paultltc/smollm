import torch

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

from m4.training.utils import END_OF_UTTERANCE_TOKEN, ASSISTANT_TOKEN, IMAGE_TOKEN

import logging
logger = logging.getLogger(__name__)

@dataclass
class SimpleDataCollatorForVisionLanguage(DataCollatorMixin):
    processor: ProcessorMixin
    tokenizer: PreTrainedTokenizerBase = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.tokenizer is None and hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer

        if self.tokenizer is None:
            raise ValueError(
                "You need to specify a tokenizer or pass it through the processor to use `DataCollatorForVisionLanguage` collators."
            )

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index(IMAGE_TOKEN)
        ]

    @property
    def eou_token_id(self) -> int:
        return self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index(END_OF_UTTERANCE_TOKEN)
        ]

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def assistant_token_ids(self) -> List[int]:
        return self.tokenizer(ASSISTANT_TOKEN, add_special_tokens=False)["input_ids"]

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = self.processor(examples)

        # If special token mask has been preprocessed, pop it from the dict.
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=batch.pop("special_tokens_mask", None)
        )

        return batch
    
    def torch_mask_tokens(
            self, 
            inputs: Any, 
            special_tokens_mask: Optional[Any] = None, 
            ignore_images: bool = True,
            ignore_query: bool = True,
            **kwargs
        ) -> Tuple[Any, Any]:
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # labels[special_tokens_mask] = -100  # We only compute loss on non-special tokens

        if ignore_images:
            # Set the image tokens to 0.0 probability if we don't want to mask them
            images_mask = labels == self.image_token_id
            labels[images_mask] = -100  # We only compute loss on masked tokens

        if ignore_query:
            query_mask = ~self._sft_completion_mask(labels)
            labels[query_mask] = -100

        return inputs, labels

    def _sft_completion_mask(
            self, 
            labels: torch.Tensor,
        ):
        """
        Create a mask for the completion part of the input sequence.
        The mask is a boolean tensor where True indicates the positions to be masked.
        """
        batch_labels = labels.tolist()

        mask = torch.zeros_like(labels, dtype=torch.long)


        for i, labels_ in enumerate(batch_labels):
            starts_ends_list = []
            start, end = None, None
            counter_eou = 0

            for idx, l_ in enumerate(labels_):
                if l_ == self.bos_token_id:
                    assert start is None and end is None, (idx, start, end)
                    # print("Start of sequence:", idx)
                elif l_ == self.eou_token_id:
                    # print("End of utterance:", idx)
                    counter_eou += 1
                    if counter_eou % 2 == 0:
                        # print(" > End of assistant.")
                        assert start is not None and end is None, (idx, start, end)
                        # Check if the next token is the assistant token
                        expected_assistant = torch.tensor(labels_[start : start + len(self.assistant_token_ids)])
                        assitant_toks = torch.tensor(self.assistant_token_ids)
                        assert torch.all(expected_assistant == assitant_toks), (idx, expected_assistant, assitant_toks)
                        end = idx - 1 
                        starts_ends_list.append((start, end))
                        start, end = None, None
                    else:   
                        # print(" > End of user.")
                        assert start is None and end is None, (idx, start, end)
                        if idx + 1 < len(labels_) and labels_[idx + 1] != self.eos_token_id:
                            start = idx + 1
                elif l_ == self.eos_token_id:
                    # print("End of sequence:", idx)
                    assert start is None and end is None, (idx, start, end)
            assert start is None and end is None, (idx, start, end)

            for start_index, end_index in starts_ends_list:
                mask[i, start_index:end_index] = 1

        return mask.bool()


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
    # mlm args
    mlm: bool = True
    mlm_probability: float = 0.5
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    # common args
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
            ignore_images: bool = True,
            ignore_query: bool = True,
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
        if ignore_images:
            images_mask = labels == self.image_token_id
            probability_matrix.masked_fill_(images_mask, value=0.0)

        # if it is sft type, set the query tokens to 0.0 probability
        if ignore_query:
            completion_mask = self._sft_completion_mask(labels)
            probability_matrix.masked_fill_(~completion_mask, value=0.0)

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