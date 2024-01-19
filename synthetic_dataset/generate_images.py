import json
import hashlib
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm import tqdm


# Read the JSONL file and return list of prompts
def read_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)['prompt']


# Generate a hash from the text prompt
def hash_prompt(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None


def generate(prompt):
    while True:
        try:
            response_vivid = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                n=1,
                style="vivid"
            )
            return response_vivid
        except Exception as ex:
            if "Rate limit" in str(ex):
                time.sleep(random.randint(30, 90))


# Generate an image using OpenAI Dalle 3 API and save the result
def generate_and_save_image(prompt):
    hash_name = hash_prompt(prompt)
    file_name = f"/Users/alekseykorshuk/PycharmProjects/image-gym/synthetic_dataset/experiments/baseline/images/{hash_name}.png"
    if os.path.exists(file_name):
        # print(f"File for prompt '{hash_name}' already exists. Skipping.")
        return

    response_vivid = generate(prompt)
    image_url = response_vivid.data[0].url
    image = download_image(image_url)
    image.save(file_name)


# Main function to orchestrate the program
def main(file_path):
    prompts = list(read_jsonl_file(file_path))
    hashes = [hash_prompt(prompt) for prompt in prompts]
    print(len(prompts))
    print(len(set(prompts)))
    print(len(hashes))
    print(len(set(hashes)))

    # Use ThreadPoolExecutor for parallel execution
    with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_and_save_image, prompt) for prompt in prompts]
            for future in as_completed(futures):
                pbar.update(1)


# Assuming the JSONL file is named 'prompts.jsonl' and located in the same directory
jsonl_file_path = "/Users/alekseykorshuk/PycharmProjects/image-gym/synthetic_dataset/experiments/baseline/accepted.jsonl"
main(jsonl_file_path)
