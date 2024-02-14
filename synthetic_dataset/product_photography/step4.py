import base64
import hashlib
import json
import os
import io
import requests
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from openai import OpenAI

client = OpenAI()

save_path = "./data/midjourney_images_step4"

os.makedirs(save_path, exist_ok=True)


def apply_1(sample):
    output_path = os.path.join(save_path, sample["id"] + ".json")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
        return {
            "decision": data["decision"],
        }
    if sample["midjourney_image"] is None:
        return {
            "decision": 0,
        }
    base64_image = encode_image(sample["midjourney_image"])
    decision = None
    for temperature in [0.3, 0.5, 0.7, 1.0]:
        try:
            data = get_json_response(base64_image, temperature)
            with open(output_path, "w") as f:
                json.dump(data, f)
            decision = data["decision"]
            break
        except Exception as e:
            print(e)
            continue

    return {
        "decision": int(decision),
    }


def get_json_response(base64_image, temperature=0.3):
    completion = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": prompt1},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=temperature
    )
    completion_content = completion.choices[0].message.content
    try:
        json_string = completion_content[completion_content.index("{"):]
        json_string = json_string[::-1][json_string[::-1].index("}"):]
        json_string = json_string[::-1]
        data = json.loads(json_string)
    except Exception as e:
        raise e
    return data


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def apply_2(sample):
    output_path = os.path.join(save_path, sample["id"] + ".png")
    if os.path.exists(output_path):
        image = Image.open(output_path)
        return {
            "image": image,
        }
    decision = sample["decision"]
    image = None
    if decision != 0:
        image = get_selected_image(sample["midjourney_image"], decision)
        image.save(output_path)
        image = Image.open(output_path)
    return {
        "image": image,
    }


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


prompt1 = """You are an expert in product photography. You will be given a collage of AI generated image. Your task is to select best image from the collage or confirm that all images are bad.


# Input:

Square image 2x2, let's label images as:
- 1: Top left
- 2: Top right
- 3: Bottom left
- 4: Bottom right


# Output:

Your response should follow JSON format with step-by-step thinking:
```json
{
    "product": "", // Define what is the product
    "thoughts_image_{LABEL}": {
        "{RULE_NUMBER}": "", // Your thoughts, criqique, evaluation if image follow the rules, think step by step
        ...
    },
    ...
    "overall_thoughts: "", // Your final observations and decision based on the per-image thoughts. Be very harsh!!!
    "decision": int, // Image label or 0 if all images are bad
}
```


# Rules:

1. Composition: The image should adhere to certain principles of professional photography composition, including the “Rule Of Thirds”, “Depth and Layering”, and more. Negative examples may include imbalance in visual weight, such as when all focal subjects are concentrated on one side of the frame, subjects captured from less flattering angles, or instances where the primary subject is obscured, or surrounding unimportant objects are distracting from the subject.

2. Lighting: You are looking for dynamic lighting with balanced exposure that enhances the image, for example, lighting that originates from an angle, casting highlights on select areas of the background and subject(s). You try to avoid artificial or lackluster lighting, as well as excessively dim or overexposed light.

3. Color and Contrast: Prefer images with vibrant colors and strong color contrast. Avoid monochromatic images or those where a single color dominates the entire frame.

4. Subject and Background: The image should have a sense of depth between the foreground and background elements. The background should be uncluttered but not overly simplistic or dull. The focused subjects must be intentionally placed within the frame, ensuring that all critical details are clearly visible without compromise. For instance, in a portrait, the primary subject of image should not extend beyond the frame or be obstructed. Furthermore, the level of detail on the foreground subject is extremely important. Additionally, product should have reasonable size on the image, so that a human can clearly identify it.

5. Hallucinations: Avoid images with hallucinations including but not limited to:
- floating objects that should be attached
- random parts of the objects that are not expected to be present
- incorrect form of the product or environment
- multiple products but expected single
-  etc.
This part is very important and if "True" should trigger rejection.

6. Subjective Assessments: Furthermore, provide your subjective assessments to ensure that only images of exceptionally aesthetic quality are retained by answering a couple of questions, such as: (i) Does this image convey a compelling story? (ii) Could it have been captured significantly better? (iii) Is this among the best photos you’ve ever seen for this particular content?

Overall, be very harsh!!!
You must better reject all images, then select one bad image!"""

if __name__ == "__main__":
    ds = load_dataset("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage")
    ds = ds.map(apply_1, num_proc=10, writer_batch_size=100)
    ds = ds.filter(lambda x: x != 0, input_columns=["decision"])
    ds = ds.map(apply_2, num_proc=10, writer_batch_size=100)
    ds = ds.remove_columns(["decision", "midjourney_image"])
    ds.push_to_hub("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage-images")
