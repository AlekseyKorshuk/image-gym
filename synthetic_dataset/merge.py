import hashlib
import os
import json
from io import BytesIO

from datasets import Dataset


from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = False


# Read the JSONL file and return list of prompts
def read_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)['prompt']


# Generate a hash from the text prompt
def hash_prompt(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


# Function to collect and merge prompt with URL
def collect_and_merge_data(prompts_file, output_folder):
    merged_data = []
    prompts = list(read_jsonl_file(prompts_file))

    test = []
    for prompt in prompts:
        hash_name = hash_prompt(prompt)
        image_file_path = os.path.join(output_folder, f"{hash_name}.png")

        if os.path.exists(image_file_path):
            file_stats = os.stat(image_file_path)
            test.append(file_stats.st_size / (1024 * 1024))
            merged_data.append({"openai_prompt": prompt, "image": Image.open(image_file_path)})
    print(set(test))
    return merged_data


# Main function
def main():
    prompts_file = "/Users/alekseykorshuk/PycharmProjects/image-gym/synthetic_dataset/experiments/baseline/accepted.jsonl"
    output_folder = "/Users/alekseykorshuk/PycharmProjects/image-gym/synthetic_dataset/experiments/baseline/images/"

    # Collect and merge data
    dataset = collect_and_merge_data(prompts_file, output_folder)

    ds = Dataset.from_list(dataset)
    print(ds)
    ds.push_to_hub("AlekseyKorshuk/product-photography-synthetic-baseline-5k")


if __name__ == "__main__":
    main()
