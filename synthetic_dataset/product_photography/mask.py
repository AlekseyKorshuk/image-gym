from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import os
import numpy as np
from PIL import Image, ImageOps
from transparent_background import Remover
import os
from functools import partial
import torch
from PIL import Image
import base64
import io

import time
import os

ds = load_dataset("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage-filtered")

os.makedirs("./data/masks", exist_ok=True)

torch.multiprocessing.set_start_method('spawn', force=True)

remover = Remover(device="cuda")


def get_map(image):
    return remover.process(image.convert('RGB'), type='map')


func = partial(get_map)


def prepare(row):
    mask_path = f"./data/masks/{row['id']}.png"
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


def encode_image(image):
    image = image.resize((512, 512)).convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


prompt = """You are an expert in prompt engineering for Text2Image models, specializing in product photography. Your role involves converting user queries into optimal prompts.
Your task is to provide ideal prompts to reproduce the given image.
Make sure to follow the rules!!!


# Prompt structure:
```
{detailed product description}, {specific product placement in the image}, {rich background and environment details}, {precise product/camera view}, {distinct styles or vibes, if any}, {any additional relevant details}
```
- Use a single, lowercase sentence.

## Product Description:
- Describe the product's appearance, including its parts and unique features.
- Enhanced Examples:
   - a gleaming mahogany brown leather briefcase with intricate stitch detailing
   - a vibrant array of hand-painted ceramic bowls in various sizes
   - a modern, minimalist desk lamp with an adjustable matte black arm and a bright LED light
   - an antique brass telescope with an extended tripod, showing intricate engravings

## Product Placement in the Image:
- Specify the product's position and what parts are visible or hidden.
- Enhanced Examples:
   - perched atop a stack of old, weathered books with faint gold lettering
   - nestled among a scatter of fresh, bright green fern leaves
   - partially draped with a delicate, sheer white linen cloth
   - positioned against a backdrop of old brick wall, bathed in warm, golden-hour sunlight

## Background and Environment:
- Provide a detailed description of the background and surroundings.
- Enhanced Examples:
   - surrounded by a blur of a busy urban street at dusk, with glowing streetlights and distant city sounds
   - set against a serene, misty mountain landscape with a hint of snow-capped peaks in the distance
   - in a bustling, colorful open-air market scene, filled with the vibrant hues of fruits and fabrics

## Product/Camera View:
- Detail the camera's perspective on the product.
- Enhanced Examples:
   - a macro, close-up view showing the texture and fine details of a hand-woven basket
   - an angled, isometric view of a high-tech gaming keyboard, highlighting its ergonomic design
   - a dynamic, three-quarter view of an electric skateboard, emphasizing both its sleek design and sturdy wheels


## Prompt Rules:
- Limit to 75 CLIP tokens, approximately 60 words, utilizing the full length but avoiding overflow.
- Ensure clarity and accuracy without introducing hallucinations or unrelated elements.


# Response structure
Respond in JSON format:
```
{
  "thoughts": "Think step by step here: what is on the image, how to write ideal prompt to fully reproduce the image and how to follow the rules and structure.",
  "prompt": ""
}
```
Think step by step in `thoughts` in your response.
"""

from openai import OpenAI
import json
from functools import lru_cache

client = OpenAI()


def get_prompt(image):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image)}",
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.0
    )
    json_string = response.choices[0].message.content
    json_string = json_string[json_string.index("{"):]
    json_string = json_string[::-1][json_string[::-1].index("}"):]
    json_string = json_string[::-1]
    data = json.loads(json_string)
    return data["prompt"]


ds = ds.shuffle(42)

os.makedirs("./data/prompts", exist_ok=True)


def prepare(row):
    path = f"./data/prompts/{row['id']}.json"
    if os.path.exists(path):
        with open(path) as f:
            result = json.load(f)
        return result

    text = None
    while True:
        try:
            text = get_prompt(row["image"])
            break
        except Exception as ex:
            if "rate" in str(ex).lower():
                time.sleep(2)
            else:
                print(ex)
                break
    result = {
        "text": text
    }

    with open(path, "w") as outfile:
        json.dump(result, outfile)
    return result


ds = ds.map(prepare, num_proc=10)
ds = ds.filter(lambda example: example is not None, input_columns=["text"])
ds.push_to_hub("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage-filtered-annotated")
