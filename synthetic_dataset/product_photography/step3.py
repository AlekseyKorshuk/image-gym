import hashlib
import json
import os

import requests
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image

save_path = "/Users/alekseykorshuk/PycharmProjects/image-gym/synthetic_dataset/product_photography/data_tiny/midjourney_images"

os.makedirs(save_path, exist_ok=True)


def apply(sample):
    output_path = os.path.join(save_path, sample['id'] + ".png")
    task_path = os.path.join(save_path, sample['id'] + ".json")
    if os.path.exists(task_path):
        with open(task_path, "r") as f:
            task = json.load(f)
    else:
        task = fetch_task(sample["task_id"])
        with open(task_path, "w") as f:
            json.dump(task, f)

    if os.path.exists(output_path):
        image = None
        if task["status"] == "finished":
            try:
                image = Image.open(output_path)
            except Exception as e:
                print(e)

        return {
            "status": task["status"],
            "midjourney_image": image
        }
    image = None
    if task["status"] == "finished":
        url = f"https://img.midjourneyapi.xyz/mj/{sample['task_id']}.png"
        response = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(response.content)
        image = Image.open(output_path)
    return {
        "status": task["status"],
        "midjourney_image": image
    }


def fetch_task(task_id):
    endpoint = "https://api.midjourneyapi.xyz/mj/v2/fetch"
    response = requests.post(endpoint, json={"task_id": task_id})
    return response.json()


if __name__ == "__main__":
    ds = load_dataset("AlekseyKorshuk/product-photography-tiny-prompts-tasks")
    ds = ds.map(apply, num_proc=10)
    ds = ds.filter(lambda x: x == "finished", input_columns=["status"])
    ds.save_to_disk("./output/step3")
