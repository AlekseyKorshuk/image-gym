import base64
import hashlib
import json
import os
import io
import requests
import tqdm
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from PIL import Image
from openai import OpenAI

save_path = "./data/midjourney_images_temp"

os.makedirs(save_path, exist_ok=True)

ds = concatenate_datasets(
    [
        load_dataset("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage", split="train"),
        load_dataset("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage", split="validation"),
    ]
)
print(ds)

with open("midjourney-v0.json") as f:
    data = json.load(f)


def get_selected_image(collage: Image, decision: int) -> Image:
    # Get the dimensions of the collage
    width, height = collage.size

    # Calculate the size of each quadrant
    quadrant_size = width // 2

    # Define the bounding box for each decision
    if decision == 1:  # Top left
        box = (0, 0, quadrant_size, quadrant_size)
    elif decision == 2:  # Top right
        box = (quadrant_size, 0, width, quadrant_size)
    elif decision == 3:  # Bottom left
        box = (0, quadrant_size, quadrant_size, height)
    elif decision == 4:  # Bottom right
        box = (quadrant_size, quadrant_size, width, height)
    else:
        raise ValueError("Decision must be between 1 and 4.")

    # Crop the image according to the box and return
    selected_image = collage.crop(box)
    return selected_image


dataset = []
for sample in tqdm.tqdm(ds):
    try:
        annotations = []
        for i in range(1, 5):
            if data[sample["id"]][f"quality_{i}"] == "GOOD":
                annotations.append(i)

        for annotation in annotations:
            image = get_selected_image(sample["midjourney_image"], annotation)
            image_path = os.path.join(save_path, f"{sample['id']}_{annotation}.png")
            image.save(image_path)
            dataset.append(
                {
                    "id": sample["id"] + "_" + str(annotation),
                    "image": Image.open(image_path),
                    "category": sample["category"],
                    "product": sample["product"],
                    "prompt": sample["prompt"],
                }
            )
    except Exception as ex:
        print(ex)

new_ds = Dataset.from_list(dataset)
new_ds.push_to_hub("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage-filtered")
