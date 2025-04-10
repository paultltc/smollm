import re

from typing import Tuple

from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig

from m4.models.idefics.configuration_idefics import IdeficsConfig
from m4.models.idefics.modeling_idefics import IdeficsForCausalLM
from m4.models.vgpt2.configuration_vgpt2 import VGPT2Config
from m4.models.vgpt2.modeling_vgpt2 import VGPT2LMHeadModel
from m4.models.vllama3.configuration_vllama3 import VLlama3Config
from m4.models.vllama3.modeling_vllama3 import VLlama3ForCausalLM, VLlama3ForMaskedLM
from m4.models.vmistral.configuration_vmistral import VMistralConfig
from m4.models.vmistral.modeling_vmistral import VMistralForCausalLM
from m4.models.vbert.modeling_vbert import VBertForMaskedLM
from m4.models.vbert.configuration_vbert import VBertConfig

from m4.training.utils import regex_lookup


model_name2classes = {
    r"gpt2": [VGPT2Config, VGPT2LMHeadModel],
    r"idefics": [IdeficsConfig, IdeficsForCausalLM],
    r"mistral": [VMistralConfig, VMistralForCausalLM],
    r"llama": [VLlama3Config, VLlama3ForCausalLM],
    r"smollm": [VLlama3Config, VLlama3ForCausalLM],
    r"smolvlm": [VLlama3Config, VLlama3ForCausalLM],
    r"bert": [VBertConfig, VBertForMaskedLM],
}


def model_name_to_classes(model_name_or_path) -> Tuple[PretrainedConfig, PreTrainedModel]:
    """returns config_class, model_class for a given model name or path"""
    return regex_lookup(model_name_or_path, model_name2classes)