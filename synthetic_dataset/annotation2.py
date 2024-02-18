import io
import json
import os

import argilla as rg
from argilla._constants import DEFAULT_API_KEY
from argilla.client.feedback.utils import image_to_html
from datasets import load_dataset, concatenate_datasets
import tqdm
from PIL import Image
import concurrent.futures

api_url = os.environ.get("ARGILLA_API_URL")
api_key = os.environ.get("ARGILLA_API_KEY")

rg.init(api_url=api_url, api_key=api_key, workspace="admin",
        extra_headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})

# Configure the FeedbackDataset
dataset = rg.FeedbackDataset(
    fields=[
        rg.TextField(name="sample", use_markdown=True, required=True),
    ],
    questions=[
        rg.LabelQuestion(
            name="quality",
            title="Is the sample good?",
            labels={"BAD": "Bad", "GOOD": "Good"},
            required=True,
            visible_labels=None
        ),
    ],
)

ds = load_dataset("AlekseyKorshuk/product-photography-midjourney-merged", split="train")
print(ds)


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def process_sample(index, ds):
    sample = ds[index]
    try:
        image = sample["image"].resize((512, 512))
        byte_buffer = io.BytesIO()
        image.save(byte_buffer, format='PNG')
        byte_string = byte_buffer.getvalue()
        record = rg.FeedbackRecord(
            fields={"sample": image_to_html(byte_string, file_type="png")},
            external_id=sample["id"],
        )
        return record
    except Exception as e:
        print(e)
        return None


def main(ds):
    records = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare for progress bar
        futures = [
            executor.submit(
                process_sample,
                index,
                ds
            )
            for index in tqdm.trange(len(ds))
        ]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(ds)):
            result = future.result()
            if result is not None:
                records.append(result)
    return records


records = main(ds)
dataset.add_records(records)
try:
    dataset.push_to_argilla(name="midjourney-v0-merged", workspace="admin")
except Exception as ex:
    print(ex)

# feedback = rg.FeedbackDataset.from_argilla("midjourney-v0-merged", workspace="admin")
#
# data = {}
# for record in tqdm.tqdm(feedback.records):
#     try:
#         data[record.external_id] = {
#             "quality": str(record.responses[0].values["quality"].value),
#         }
#     except Exception as ex:
#         print(ex)
#
# with open("midjourney-v0-2.json", "w") as f:
#     json.dump(data, f)
