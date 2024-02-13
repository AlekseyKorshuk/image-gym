import hashlib
import json
import os

from datasets import Dataset, DatasetDict
from openai import OpenAI

client = OpenAI()
input_path = "./input.csv"

output_path = "./data/prompts"

os.makedirs(output_path, exist_ok=True)


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


prompt1 = """You are an expert in Prompt Engineering for Product Photography. You are helping to build a diverse dataset of prompts for product photographies. You will be given a product from user and your task is to response in JSON format in the following way to come up with self-contained diverse dataset of different prompts from simple to complicated, for very different ideas. Make sure to follow the RULES!!!

# Response Format:

```json
{
    "thoughts_and_planning": "Explain your thoughts and planning for the prompts, how you will approach the task, etc.",
    "train_prompts": [
        // List of 15 train prompts with diverse complexity and ideas
    ],
    "test_prompts": [
        // List of 2 test prompts to validate training
    ]
}
```

## Notes:
- East/Medium/Hard does NOT mean to write short or long prompts -- utilise all possible tokens! It means the complexity of generating such image for AI!
- Be creative! You can place products under ocean, in space, on the moon, etc. -- be creative!
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
- Product should be clearly visible on the image, consider this when creating the composition.


# Product Photography Rules:

1. Composition: The image should adhere to certain principles of professional photography composition, including the “Rule Of Thirds”, “Depth and Layering”, and more. Negative examples may include imbalance in visual weight, such as when all focal subjects are concentrated on one side of the frame, subjects captured from less flattering angles, or instances where the primary subject is obscured, or surrounding unimportant objects are distracting from the subject.

2. Lighting: You are looking for dynamic lighting with balanced exposure that enhances the image, for example, lighting that originates from an angle, casting highlights on select areas of the background and subject(s). You try to avoid artificial or lackluster lighting, as well as excessively dim or overexposed light.

3. Color and Contrast: Prefer images with vibrant colors and strong color contrast. Avoid monochromatic images or those where a single color dominates the entire frame.

4. Subject and Background: The image should have a sense of depth between the foreground and background elements. The background should be uncluttered but not overly simplistic or dull. The focused subjects must be intentionally placed within the frame, ensuring that all critical details are clearly visible without compromise. For instance, in a portrait, the primary subject of image should not extend beyond the frame or be obstructed. Furthermore, the level of detail on the foreground subject is extremely important. Additionally, product should have reasonable size on the image, so that a human can clearly identify it."""

prompt2 = """Critique your response in plain text and plan how to improve it the dataset:
- Is the dataset diverse enough in terms of ideas?
- Is the dataset diverse enough in terms of complexity?
- Are the prompts clear and concise? Do they follow all the prompt rules and structure?
- Are the prompts follow the rules of good product photography: Composition, Lighting, Color and Contrast, Subject and Background?
- Is the product single (if not specified otherwise)?

Critique in details and refine your response with the same JSON format as before!
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
    ds.push_to_hub("AlekseyKorshuk/product-photography-v1-tiny-prompts")
