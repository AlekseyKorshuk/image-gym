import argparse
import functools
import gc
import itertools
import math
import os
import random
from pathlib import Path

from packaging import version
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer, PretrainedConfig
import torchvision.transforms as T
from diffusers.training_utils import EMAModel, compute_snr

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_xformers_available

# Will error if the minimal version of diffusers is not installed.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

generation_params = {
    "strength_1": {
        "num_inference_steps": 25,
        "num_images_per_prompt": 1,
        "guidance_scale": 10.0,
        "strength": 1.0,
    },
    "strength_99": {
        "num_inference_steps": 25,
        "num_images_per_prompt": 1,
        "guidance_scale": 10.0,
        "strength": 0.99,
    },
}


def prepare_mask_and_masked_image_v0(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)
    return mask, masked_image


def prepare_mask_and_masked_image_v1(image, mask):
    masked_image = get_preprocessed_image(image, mask)
    masked_image = np.array(masked_image.convert("RGB"))
    masked_image = masked_image[None].transpose(0, 3, 1, 2)
    masked_image = torch.from_numpy(masked_image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    return mask, masked_image


def get_preprocessed_image(initial_image, mask_image):
    initial_image_size = initial_image.size
    noise_image = np.random.normal(loc=127.5, scale=127.5, size=(initial_image_size[1], initial_image_size[0], 3))
    noise_image_clipped = np.clip(noise_image, 0, 255).astype(np.uint8)
    noise_image_pil = Image.fromarray(noise_image_clipped)
    transparent_image = initial_image.copy()
    mask_array = np.array(mask_image.convert("L"))
    alpha_channel = np.where(mask_array == 255, 0, 255).astype(np.uint8)
    transparent_image.putalpha(Image.fromarray(alpha_channel))
    noise_image_rgba = noise_image_pil.convert("RGBA")
    transparent_image_rgba = transparent_image.convert("RGBA")
    combined_image = Image.alpha_composite(noise_image_rgba, transparent_image_rgba)
    return combined_image


prepare_mask_and_masked_image = prepare_mask_and_masked_image_v0


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        required=False,
        help="Path to pretrained vae or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset on the hub.",
    )
    parser.add_argument(
        "--validation_dataset_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset on the hub.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--eval_every_epoch",
        type=int,
        default=1,
        help="The number of epochs to wait before evaluating the model.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        default=False,
        help="Whenever to train text encoder",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=1e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="The minimum learning rate as a ratio of the initial learning rate.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder"
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
             "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
             "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--noise_offset", type=float, default=0,
                        help="The scale of noise offset.")  # 0.1 for Emu paper
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_path is None:
        raise ValueError("You must specify a train data directory.")

    return args


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(input_ids_1, input_ids_2, text_encoders, tokenizers, is_train=True):
    prompt_embeds_list = []

    for text_input_ids, text_encoder in zip([input_ids_1, input_ids_2], text_encoders):
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def run_generation(pipe, eval_dataset, generation_params, seed=None):
    function = functools.partial(generate, pipe=pipe, generation_params=generation_params, seed=seed)
    result_ds = eval_dataset.map(function, batched=True, batch_size=2, num_proc=1)
    return [
        wandb.Image(sample["generated_image"], caption=f'{i}: {sample["text"]}')
        for i, sample in enumerate(result_ds)
    ]


def generate(batch, pipe, generation_params, seed=None):
    negative_prompt = "watermark, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, " \
                      "morbid, mutilated, mutation, deformed, dehydrated, bad anatomy, bad proportions, " \
                      "floating object, levitating, hallucination"

    preprocess = transforms.Compose(
        [
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(1024),
        ]
    )

    pairs = [
        prepare_mask_and_masked_image(
            preprocess(image),
            preprocess(image_mask)
        )
        for image, image_mask in zip(batch["image"], batch["image_mask"])
    ]
    images = pipe(
        prompt=batch["text"],
        negative_prompt=negative_prompt,
        generator=torch.manual_seed(seed if seed is not None else torch.seed()),
        image=[pair[1] for pair in pairs],
        mask_image=[pair[0] for pair in pairs],
        **generation_params
    ).images
    return {
        "generated_image": images
    }


def _get_cosine_schedule_with_min_lr_lambda(
        current_step: int,
        *,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float
):
    # Warm up
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    # Cosine learning rate decay
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    scaling = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (1 - min_lr_ratio) * scaling + min_lr_ratio


def get_cosine_schedule_with_min_lr(
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.0,
):
    """
    Create a learning rate schedule which has:
        - linear warmup from 0 -> `max_lr` over `num_warmup_steps`
        - cosine learning rate annealing from `max_lr` -> `min_lr` over `num_training_steps`
    """

    lr_lambda = functools.partial(
        _get_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
        in_channels=9,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False
    )
    vae.requires_grad_(False)

    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
            in_channels=9,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config,
                            inv_gamma=1.0,
                            power=3 / 4)
        ema_unet.to(accelerator.device)
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    unet_parameters_with_lr = {"params": unet.parameters(), "lr": args.learning_rate}
    params_to_optimize = [unet_parameters_with_lr]
    if args.train_text_encoder:
        text_encoder_one.train()
        text_encoder_two.train()
        params_to_optimize.extend(
            [
                {
                    "params": text_encoder_one.parameters(),
                    "weight_decay": args.adam_weight_decay_text_encoder
                    if args.adam_weight_decay_text_encoder
                    else args.adam_weight_decay,
                    "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
                },
                {
                    "params": text_encoder_two.parameters(),
                    "weight_decay": args.adam_weight_decay_text_encoder
                    if args.adam_weight_decay_text_encoder
                    else args.adam_weight_decay,
                    "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
                }
            ]
        )
    else:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warn(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warn(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(batch, text_encoders, tokenizers, is_train=True):
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            batch["input_ids_1"], batch["input_ids_2"], text_encoders, tokenizers, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.repeat(len(batch["input_ids_1"]), 1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    with accelerator.main_process_first():
        eval_dataset = load_dataset(args.validation_dataset_path)
        eval_dataset = eval_dataset["validation"] if "validation" in eval_dataset.keys() else eval_dataset["train"]
        train_dataset = load_dataset(args.dataset_path)
        if "validation" in list(train_dataset.keys()):
            validation_dataset = train_dataset["validation"]
            train_dataset = train_dataset["train"]
        else:
            train_dataset = load_dataset(args.dataset_path, split="train[:95%]")
            validation_dataset = load_dataset(args.dataset_path, split="train[95%:]")

    # del text_encoders, tokenizers
    gc.collect()
    torch.cuda.empty_cache()

    def prepare_train_dataset(dataset, accelerator):
        resize_and_crop_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
            ]
        )
        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(examples):
            images = [resize_and_crop_transforms(image.convert("RGB")) for image in examples["image"]]
            mask_images = [resize_and_crop_transforms(image.convert("RGB")) for image in examples["image_mask"]]
            pairs = [
                prepare_mask_and_masked_image(pil_image, mask_image)
                for pil_image, mask_image in zip(images, mask_images)
            ]
            examples["pixel_values"] = [image_transforms(image) for image in images]
            examples["instance_images"] = [pair[1] for pair in pairs]
            examples["instance_masks"] = [pair[0] for pair in pairs]

            return examples

        with accelerator.main_process_first():
            dataset = dataset.with_transform(preprocess_train)

        return dataset

    train_dataset = prepare_train_dataset(train_dataset, accelerator)
    validation_dataset = prepare_train_dataset(validation_dataset, accelerator)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        masked_images = torch.stack([example["instance_images"] for example in examples])
        masked_images = masked_images.to(memory_format=torch.contiguous_format).float()

        # prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

        # add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
        # add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])
        input_ids_1 = tokenizer_one(
            [example["text"] for example in examples],
            padding="max_length",
            max_length=tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        input_ids_2 = tokenizer_two(
            [example["text"] for example in examples],
            padding="max_length",
            max_length=tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return {
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
            # "ids": [example["id"] for example in examples],
            # "text": [example["text"] for example in examples],
            "masks": torch.stack([example["instance_masks"] for example in examples]),
            "pixel_values": pixel_values,
            "masked_images": masked_images,
            # "prompt_ids": prompt_ids,
            # "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        }

    train_dataset = ShufflerIterDataPipe(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
        drop_last=True
    )
    validation_dataset = ShufflerIterDataPipe(validation_dataset)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
        drop_last=False
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    #     num_training_steps=args.max_train_steps * accelerator.num_processes,
    # )

    lr_scheduler = get_cosine_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    # print(f"Before accelerate prepare: {train_dataloader.dataset._shuffle_enabled}")
    unet, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )
    # print(f"After accelerate prepare: {train_dataloader.dataset._shuffle_enabled}")
    torch.utils.data.graph_settings.apply_shuffle_settings(train_dataloader.dataset, shuffle=True)

    if args.use_ema:
        ema_unet.to(accelerator.device)
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    evaluation(accelerator, args, validation_dataloader, eval_dataset, vae, unet,
               ema_unet if args.use_ema else None,
               text_encoders, tokenizers, noise_scheduler, compute_embeddings, weight_dtype, global_step)
    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        step = -1
        for batch in train_dataloader:
            step += 1
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                # print("IDS:", batch["ids"])

                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Convert masked images to latent space
                masked_latents = vae.encode(
                    batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                masks = batch["masks"]
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                        for mask in masks
                    ]
                )
                mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                embeddings_dict = compute_embeddings(batch, text_encoders, tokenizers)
                unet_added_conditions = {
                    "text_embeds": embeddings_dict["text_embeds"],
                    "time_ids": embeddings_dict["time_ids"]
                }
                # Predict the noise residual
                noise_pred = unet(
                    latent_model_input, timesteps,
                    encoder_hidden_states=embeddings_dict["prompt_embeds"],
                    added_cond_kwargs=unet_added_conditions,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (unet.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "epoch": epoch + (step + 1) / len(train_dataloader),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # use args.eval_every_epoch
        if (epoch + 1) % args.eval_every_epoch == 0:
            evaluation(accelerator, args, validation_dataloader, eval_dataset, vae, unet,
                       ema_unet if args.use_ema else None,
                       text_encoders, tokenizers, noise_scheduler, compute_embeddings, weight_dtype, global_step)
            accelerator.wait_for_everyone()
            unet.train()
    accelerator.wait_for_everyone()
    # Create the pipeline using the trained modules and save it.
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            unet=unet,
            text_encoder=accelerator.unwrap_model(text_encoder_one),
            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*", "checkpoint-*", "checkpoint_*"],
                token=args.hub_token
            )

    accelerator.end_training()


def evaluation(accelerator, args, validation_dataloader, eval_dataset, vae, unet, ema_unet, text_encoders, tokenizers,
               noise_scheduler, compute_embeddings, weight_dtype, global_step):
    if accelerator.is_main_process:
        if global_step == 0:
            accelerator.log({
                "original": [
                    wandb.Image(sample["image"], caption=f'{i}: {sample["text"]}')
                    for i, sample in enumerate(eval_dataset)
                ]
            }, step=global_step)
        with torch.cuda.amp.autocast():
            if args.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoders[0]),
                text_encoder_2=accelerator.unwrap_model(text_encoders[1]),
            )
            if args.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()
            accelerator.log({
                f"seed_0_{media_title}": run_generation(pipeline, eval_dataset, gen_params, seed=0)
                for media_title, gen_params in generation_params.items()
            }, step=global_step)
            accelerator.log({
                f"random_seed_{media_title}": run_generation(pipeline, eval_dataset, gen_params, seed=None)
                for media_title, gen_params in generation_params.items()
            }, step=global_step)
            if args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())
    loss = 0
    for batch in tqdm(validation_dataloader, desc="Validation"):
        with torch.no_grad():
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            masked_latents = vae.encode(
                batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
            ).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor

            masks = batch["masks"]
            mask = torch.stack(
                [
                    torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                    for mask in masks
                ]
            )
            mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

            embeddings_dict = compute_embeddings(batch, text_encoders, tokenizers)
            unet_added_conditions = {
                "text_embeds": embeddings_dict["text_embeds"],
                "time_ids": embeddings_dict["time_ids"]
            }
            noise_pred = unet(
                latent_model_input, timesteps,
                encoder_hidden_states=embeddings_dict["prompt_embeds"],
                added_cond_kwargs=unet_added_conditions,
            ).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.snr_gamma is None:
                loss += F.mse_loss(noise_pred.float(), target.float(), reduction="mean").detach().item()
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss_ = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss_ = loss_.mean(dim=list(range(1, len(loss_.shape)))) * mse_loss_weights
                loss += loss_.mean().detach().item()

    if accelerator.is_main_process:
        loss /= len(validation_dataloader)
        logger.info(f"Validation loss: {loss}")
        accelerator.log({"val_loss": loss}, step=global_step)


if __name__ == "__main__":
    main()