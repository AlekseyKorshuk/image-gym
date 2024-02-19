from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import os
import numpy as np
from PIL import Image, ImageOps
from transparent_background import Remover
import os
from functools import partial
import torch

ds = load_dataset("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage-filtered")

os.makedirs("./data/masks", exist_ok=True)

torch.multiprocessing.set_start_method('spawn', force=True)

remover = Remover(device="cuda")


def get_map(image):
    return remover.process(image.convert('RGB'), type='map')


func = partial(get_map)


def prepare(row):
    mask_path = f"/content/masks/{row['id']}.png"
    if os.path.exists(mask_path):
        return {
            "image_mask": Image.open(mask_path),
        }
    try:
        image_mask = func(row["image"])
        image_mask = ImageOps.invert(image_mask)
        image_mask.save(mask_path)
        return {
            "image_mask": Image.open(mask_path),
        }
    except Exception as ex:
        print(ex)
        return {
            "image_mask": None,
        }


num_proc = None  # os.cpu_count()
ds = ds.map(prepare, num_proc=num_proc)
