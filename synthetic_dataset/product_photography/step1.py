import hashlib
import json
import os

from datasets import Dataset, DatasetDict
from openai import OpenAI

client = OpenAI()
output_path = "/Users/alekseykorshuk/PycharmProjects/image-gym/synthetic_dataset/product_photography/data_tiny/prompts"
input_path = "/Users/alekseykorshuk/PycharmProjects/image-gym/synthetic_dataset/product_photography/input.csv"


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def generate_prompts(sample):
    output_file = os.path.join(output_path, hash_text(sample['category'] + sample['product']) + ".json")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return json.load(f)
    messages = [
        {"role": "system", "content": prompt1},
        {"role": "user", "content": f"Category: {sample['category']}\nObject: {sample['product']}"},
    ]
    completion = client.chat.completions.create(model="gpt-4-turbo-preview", messages=messages, temperature=0.7)
    messages.append({"role": "assistant", "content": completion.choices[0].message.content})
    messages.append({"role": "user", "content": prompt2})
    completion = client.chat.completions.create(model="gpt-4-turbo-preview", messages=messages, temperature=0.7)
    completion_content = completion.choices[0].message.content
    json_string = completion_content[completion_content.index("{"):]
    json_string = json_string[::-1][json_string[::-1].index("}"):]
    json_string = json_string[::-1]
    data = json.loads(json_string)
    result = {
        "train_prompts": data["train_prompts"],
        "test_prompts": data["test_prompts"],
    }
    with open(output_file, "w") as f:
        json.dump(result, f)
    return result


prompt1 = """You are an expert in Prompt Engineering for Product Photography. You are helping to build a diverse dataset of prompts for product photographies. You will be given a category and product from user and your task is to response in JSON format in the following way to come up with self-contained diverse dataset of different prompts from simple to complicated, for very different and diverse ideas. Make sure to follow the RULES!!!

# Response Format:

```json
{
    "thoughts_and_planning": "Explain your thoughts and planning for the prompts, how you will approach the task, etc.",
    "train_prompts": [
        // List of 5 train prompts with diverse complexity and ideas
    ],
    "test_prompts": [
        // List of 1 test prompts to validate training
    ]
}
```

## Notes:
- East/Medium/Hard does NOT mean to write short or long prompts -- utilise all possible tokens! It means the complexity of generating such image for AI!
- Be creative!
- Make sure it is product photography like for advertising, not some random photo!


# How to write the prompt

## Prompt structure:
```
{detailed product description}, {specific product placement in the image}, {rich background and environment details}, {precise product/camera view}, {distinct styles or vibes, if any}, {any additional relevant details}
```
- Use a single, lowercase sentence.

### Product Description:
- Describe the product's appearance, including its parts and unique features.
- Enhanced Examples:
   - a gleaming mahogany brown leather briefcase with intricate stitch detailing
   - a vibrant array of hand-painted ceramic bowls in various sizes
   - a modern, minimalist desk lamp with an adjustable matte black arm and a bright LED light
   - an antique brass telescope with an extended tripod, showing intricate engravings

### Product Placement in the Image:
- Specify the product's position and what parts are visible or hidden.
- Enhanced Examples:
   - perched atop a stack of old, weathered books with faint gold lettering
   - nestled among a scatter of fresh, bright green fern leaves
   - partially draped with a delicate, sheer white linen cloth
   - positioned against a backdrop of old brick wall, bathed in warm, golden-hour sunlight

### Background and Environment:
- Provide a detailed description of the background and surroundings.
- Enhanced Examples:
   - surrounded by a blur of a busy urban street at dusk, with glowing streetlights and distant city sounds
   - set against a serene, misty mountain landscape with a hint of snow-capped peaks in the distance
   - in a bustling, colorful open-air market scene, filled with the vibrant hues of fruits and fabrics

### Product/Camera View:
- Detail the camera's perspective on the product.
- Enhanced Examples:
   - a macro, close-up view showing the texture and fine details of a hand-woven basket
   - an angled, isometric view of a high-tech gaming keyboard, highlighting its ergonomic design
   - a dynamic, three-quarter view of an electric skateboard, emphasizing both its sleek design and sturdy wheels

### Prompt Rules:
- Limit to 75 CLIP tokens, approximately 60 words, utilizing the full length but avoiding overflow.
- Ensure clarity and accuracy without introducing hallucinations or unrelated elements.
- Product should be clearly visible on the image, consider this when creating the composition."""

prompt2 = """Critique your response and plan how to improve it the dataset:
- Is the dataset diverse enough in terms of ideas?
- Is the dataset diverse enough in terms of complexity?
- Are the prompts clear and concise? Do they follow all the rules and structure?
- Is the product single (if not specified otherwise)?

After critique, refine your response following the same JSON format as before!
"""


def postprocess_dataset(ds):
    data = {
        "train": [],
        "validation": [],
    }
    for sample in ds:
        for prompt in sample["train_prompts"]:
            data["train"].append(
                {
                    "product_id": hash_text(sample["category"] + sample["product"]),
                    "id": hash_text(prompt),
                    "category": sample["category"],
                    "product": sample["product"],
                    "prompt": prompt,
                }
            )
        for prompt in sample["test_prompts"]:
            data["validation"].append(
                {
                    "product_id": hash_text(sample["category"] + sample["product"]),
                    "id": hash_text(prompt),
                    "category": sample["category"],
                    "product": sample["product"],
                    "prompt": prompt,
                }
            )

    dataset = DatasetDict(
        {
            split_name: Dataset.from_list(samples)
            for split_name, samples in data.items()
        }
    )
    return dataset


if __name__ == "__main__":
    ds = Dataset.from_csv(input_path)
    ds = ds.rename_columns({"Category": "category", "Product": "product"})
    ds = ds.map(generate_prompts, num_proc=10)
    ds = postprocess_dataset(ds)
    print(ds)
    ds.push_to_hub("AlekseyKorshuk/product-photography-tiny-prompts")
