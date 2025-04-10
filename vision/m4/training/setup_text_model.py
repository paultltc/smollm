import re

from typing import Tuple

from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig

from m4.training.utils import regex_lookup

# map to check the supported cv archs and also how to extract the model - in some arch, we want to
# go through a specific prefix to get to the model as in `model.vision_model` for clip
language_model_name2model = {
    r"eurobert": lambda model: model.model,
    r"modernbert": lambda model: model.model,
    r"bert": lambda model: model.bert,
}

def get_language_model(config):
    """returns the language model for a given config"""
    language_model_name = config.text_config.text_model_name

    language_model_config = AutoConfig.from_pretrained(language_model_name, trust_remote_code=True)

    model = AutoModel.from_config(language_model_config, trust_remote_code=True)

    extractor = regex_lookup(language_model_name, language_model_name2model)
    return extractor(model)