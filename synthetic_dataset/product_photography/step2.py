import hashlib
import json
import os
import random

import requests
from datasets import Dataset, DatasetDict, load_dataset

save_path = "./data/midjourney"

os.makedirs(save_path, exist_ok=True)


def apply(sample):
    output_path = os.path.join(save_path, sample['id'] + ".json")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return {
                "task_id": json.load(f)["task_id"]
            }
    response = midjourney_imagine(sample["prompt"])
    with open(output_path, "w") as f:
        json.dump(response, f)
    return {
        "task_id": response["task_id"],
    }


def midjourney_imagine(prompt, version="6.0"):
    endpoint = "https://api.midjourneyapi.xyz/mj/v2/imagine"
    headers = {"X-API-KEY": os.environ.get("MIDJOURNEY_API_KEY")}
    style = random.choice([100, 250, 500, 750])
    data = {
        "prompt": prompt.strip() + f", hyper-realistic product photography --s {style} --v {version}",
        "aspect_ratio": "1:1",
        "process_mode": "fast",
        "webhook_endpoint": "",
        "webhook_secret": ""
    }
    response = requests.post(endpoint, headers=headers, json=data)
    return response.json()


if __name__ == "__main__":
    ds = load_dataset("AlekseyKorshuk/product-photography-v1-tiny-prompts")
    ds = ds.map(apply)
    ds.push_to_hub("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks")
