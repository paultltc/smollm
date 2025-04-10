import json
import logging
import os
import sys
import time
from datetime import timedelta

import accelerate
import datasets
import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from peft import LoraConfig, PeftConfig
from torch.profiler.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from transformers import AddedToken  # AddedToken is needed for the eval of the tokenizer params # noqa: F401
from transformers import AutoTokenizer  # noqa: F401
from transformers.utils import ContextManagers, is_torch_tf32_available

import m4
from m4.training.config import get_config
from m4.training.dataset import get_dataloaders
from m4.training.trainer import Trainer
from m4.training.enums import DatasetNames
from m4.training.utils import VisionEncoderTypes, accelerate_torch_dtype, build_image_transform
from m4.utils.progress import M4_DISABLE_RICH
from m4.utils.training.timer import Timer, format_secs_to_time


logging.basicConfig(
    level=logging.INFO,
    format=" - %(process)d - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

if not M4_DISABLE_RICH:
    from rich.logging import RichHandler

    logging.getLogger("").addHandler(RichHandler(level=logging.INFO))

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    START_TIME = time.time()

    # this gives a very nice speed boost on Ampere
    if is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    config = get_config()

    # @TEMPORARY GATE -- if resuming, `realtime_processing` must be True
    if config.hparams.resume_run and not config.data_param.realtime_processing:
        raise NotImplementedError("Instant resume functionality not yet supported for non-iterable datasets!")

    # Initialize accelerator
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=config.hparams.timeout))]
    accelerator = Accelerator(
        log_with="all",
        rng_types=["torch", "cuda", "generator"],
        gradient_accumulation_steps=config.hparams.grad_acc_size,
        kwargs_handlers=kwargs_handlers,
    )

    if config.hparams.timing_break_down:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            timer1 = Timer()
            time_deltas = {}
            timer1.start()

    # logger behavior - this and sub-systems
    main_process_log_level = m4.utils.logging.get_log_levels_dict()[os.getenv("M4_VERBOSITY", "info")]
    log_level = main_process_log_level if accelerator.is_main_process else logging.ERROR
    m4.utils.logging.set_verbosity(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    if AcceleratorState().deepspeed_plugin is not None:
        from deepspeed.utils import logger as ds_logger
        ds_logger.setLevel(log_level)

    if config.hparams.kill_switch_path is not None and config.hparams.kill_switch_path.exists():
        logger.info("** Kill switch activated. Exiting the training before it even starts. **")
        sys.exit()

    accelerate.utils.set_seed(config.hparams.seed)

    logger.info(f"** The job is running with the following arguments: **\n{config}\n **** ")

    # Load model
    vl_model = config.load_model(torch_dtype=accelerate_torch_dtype())

    # If we want to use_lora and are starting a pretraining, or if we want to use a new lora for fine tuning, create the config and add the adapter.
    if (
        config.hparams.use_lora
        and not config.hparams.resume_run
        and not (config.hparams.is_fine_tuning and config.hparams.lora_name is not None)
    ):
        # Identify the target_modules with the patterns_to_loraify given in config.
        target_modules = []
        for name, param in vl_model.named_parameters():
            patterns_to_loraify_in_name = [
                all(pattern in name for pattern in pattern_list) for pattern_list in config.hparams.patterns_to_loraify
            ]
            if any(patterns_to_loraify_in_name):
                # Take off the suffixes ".weight" or ".bias"
                target_module_name = ".".join(name.split(".")[:-1])
                target_modules.append(target_module_name)
        peft_config = LoraConfig(
            target_modules=target_modules,
            **config.hparams.lora_config,
        )
        vl_model.add_adapter(peft_config)
        vl_model.enable_adapters()
        logger.info("Loaded new adapter")

    # If the model has a lora, we want to unfreeze some layers which got frozen when loading the lora
    if config.hparams.use_lora:
        for name, param in vl_model.named_parameters():
            patterns_to_unfreeze_in_name = [
                all(pattern in name for pattern in pattern_list)
                for pattern_list in config.hparams.patterns_to_unfreeze
            ]
            if any(patterns_to_unfreeze_in_name):
                param.requires_grad_(True)

    # Get the seq_len for a single image as it is necesssary for packing
    single_image_seq_len = (
        vl_model.config.perceiver_config.resampler_n_latents
        if hasattr(vl_model.config, "use_resampler") and vl_model.config.use_resampler
        else int(((vl_model.config.vision_config.image_size // vl_model.config.vision_config.patch_size) ** 2) / (vl_model.config.pixel_shuffle_factor**2))
        # else (vl_model.config.vision_config.image_size // vl_model.config.vision_config.patch_size) ** 2
    )

    if config.hparams.timing_break_down:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            time_deltas["model_load"] = timer1.delta()

    # Load tokenizer
    tokenizer = config.get_tokenizer(model_vocab_size=vl_model.config.vocab_size)
    # tokenizer.pad_token_id = 2      # TODO: Why is this necessary? Should be handled by the tokenizer itself

    if config.hparams.timing_break_down:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            time_deltas["tokenizer_load"] = timer1.delta()

    vision_model_name = vl_model.config.vision_config.vision_model_name
    vision_encoder_type = None
    for encoder_type in VisionEncoderTypes:
        if encoder_type.value in vision_model_name.lower():
            vision_encoder_type = encoder_type
            break
    train_image_transforms = {}
    val_image_transforms = {}
    for dataset_name in DatasetNames:
        dataset_param = getattr(config.data_param, dataset_name.value)
        setattr(dataset_param, "vision_encoder_max_image_size", vl_model.config.vision_config.image_size)
        train_image_transform = build_image_transform(
            max_image_size=vl_model.config.vision_config.image_size,
            min_image_size=dataset_param.min_image_size,
            image_size=None,
            vision_encoder_type=vision_encoder_type,
            dataset_name=dataset_name,
            scale_up_max=dataset_param.scale_up_max,
            scale_up_frequency=dataset_param.scale_up_frequency,
        )
        train_image_transforms[dataset_name.name.lower()] = train_image_transform
        val_image_transform = build_image_transform(
            max_image_size=vl_model.config.vision_config.image_size,
            min_image_size=dataset_param.min_image_size,
            image_size=None,
            eval=True,
            vision_encoder_type=vision_encoder_type,
            dataset_name=dataset_name,
        )
        val_image_transforms[dataset_name.name.lower()] = val_image_transform

    # Initialize data loaders
    if accelerator.is_local_main_process:
        train_loader, val_loader = get_dataloaders(
            config,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            tokenizer=tokenizer,
            train_image_transforms=train_image_transforms,
            val_image_transforms=val_image_transforms,
            image_seq_len=single_image_seq_len,
        )
        if config.hparams.loss_weights_per_dataset is not None:
            if config.hparams.grad_acc_size % train_loader.dataset.num_datasets != 0:
                raise ValueError(
                    "grad_acc_size must be a multiple of num_datasets when accumulating the loss over datasets"
                )
            if config.hparams.loss_weights_per_dataset is not None:
                if train_loader.dataset.num_datasets != len(config.hparams.loss_weights_per_dataset):
                    raise ValueError(
                        "num_datasets must equal length of loss_weights_per_dataset when accumulating the loss over"
                        " datasets"
                    )
    accelerator.wait_for_everyone()

    # And then send it to the rest of them
    if not accelerator.is_local_main_process:
        train_loader, val_loader = get_dataloaders(
            config,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            tokenizer=tokenizer,
            train_image_transforms=train_image_transforms,
            val_image_transforms=val_image_transforms,
            image_seq_len=single_image_seq_len,
        )
    accelerator.wait_for_everyone()

    if config.hparams.timing_break_down:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            time_deltas["dl_load"] = timer1.delta()

    # If the sole purpose of the job is to pre-process the dataset, exit here.
    if config.hparams.just_preprocess:
        logger.info("Preprocessing finished. Exiting the job.")
        sys.exit()

    # Get max_num_tokens
    try:
        config.hparams.max_num_tokens = len(train_loader.dataset) * config.data_param.max_seq_len
    except TypeError:
        # Can't have max_num_tokens because it is an IterableDataset
        config.hparams.max_num_tokens = -1

    # Saving config after it has been auto-populated
    config.save_config_state()

    trainer = Trainer(
        accelerator=accelerator,
        vl_model=vl_model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    if config.hparams.timing_break_down:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            time_deltas["trainer_load"] = timer1.delta()
            time_deltas["all_load"] = timer1.elapsed()
            timer1.stop()

            # Finalize
            print(f"""
    [TIME] Model:      {format_secs_to_time(time_deltas["model_load"])}
    [TIME] Tokenizer:  {format_secs_to_time(time_deltas["tokenizer_load"])}
    [TIME] DataLoader: {format_secs_to_time(time_deltas["dl_load"])}
    [TIME] Trainer:    {format_secs_to_time(time_deltas["trainer_load"])}
    [TIME] Total load: {format_secs_to_time(time_deltas["all_load"])}
            """)

    maybe_torch_profile = []
    if config.hparams.use_torch_profiler and accelerator.is_main_process:
        torch_profiler_export_path = config.hparams.save_dir / "torch_profiler"
        maybe_torch_profile_scheduler = torch.profiler.schedule(
            skip_first=10,
            wait=5,
            warmup=1,
            active=2,
            # repeat=2
        )
        maybe_torch_profile.append(
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=tensorboard_trace_handler(torch_profiler_export_path),
                record_shapes=False,
                profile_memory=True,
                with_stack=True,
                schedule=maybe_torch_profile_scheduler,
            )
        )

    with ContextManagers(maybe_torch_profile):
        train_logs = trainer.train(maybe_torch_profile[0] if len(maybe_torch_profile) == 1 else None)

    if accelerator.is_main_process:
        logger.info(f"Last step directory: {trainer.last_opt_step_dir}")
        logger.info(f"Training logs: {train_logs}")

        train_log_file = config.hparams.save_dir / "train_logs.json"
        with open(train_log_file, "w") as fh:
            json.dump(train_logs, fh)

        print(f"LOSS: {train_logs.get('per_token_loss', 0)}")
