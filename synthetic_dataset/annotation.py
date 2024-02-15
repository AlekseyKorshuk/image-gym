import io
import json
import os

import argilla as rg
from argilla._constants import DEFAULT_API_KEY
from argilla.client.feedback.utils import image_to_html
from datasets import load_dataset, concatenate_datasets
import tqdm
from PIL import Image

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
            name="quality_1",
            title="Is the sample [1] good?",
            labels={"BAD": "Bad", "GOOD": "Good"},
            required=True,
            visible_labels=None
        ),
        rg.LabelQuestion(
            name="quality_2",
            title="Is the sample [2] good?",
            labels={"BAD": "Bad", "GOOD": "Good"},
            required=True,
            visible_labels=None
        ),
        rg.LabelQuestion(
            name="quality_3",
            title="Is the sample [3] good?",
            labels={"BAD": "Bad", "GOOD": "Good"},
            required=True,
            visible_labels=None
        ),
        rg.LabelQuestion(
            name="quality_4",
            title="Is the sample [4] good?",
            labels={"BAD": "Bad", "GOOD": "Good"},
            required=True,
            visible_labels=None
        ),
    ],
)

ds = concatenate_datasets(
    [
        load_dataset("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage", split="train"),
        load_dataset("AlekseyKorshuk/product-photography-v1-tiny-prompts-tasks-collage", split="validation"),
    ]
)
print(ds)


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


records = []
for i, sample in tqdm.tqdm(enumerate(ds), total=len(ds)):
    # if i == 10:
    #     break
    try:
        image = sample["midjourney_image"].resize((1024, 1024))
        byte_buffer = io.BytesIO()
        image.save(byte_buffer, format='PNG')
        byte_string = byte_buffer.getvalue()
        records.append(
            rg.FeedbackRecord(
                fields={
                    "sample": image_to_html(byte_string, file_type="png"),
                },
                external_id=sample["id"],
            ),
        )
    except Exception as e:
        print(e)
        continue
dataset.add_records(records)
try:
    dataset.push_to_argilla(name="midjourney-v0", workspace="admin")
except Exception as ex:
    print(ex)

# feedback = rg.FeedbackDataset.from_argilla("vivid-v0", workspace="admin")
#
# data = []
# for record in tqdm.tqdm(feedback.records):
#     data.append(
#         {
#             "id": record.external_id,
#             "quality": str(record.responses[0].values["quality"].value),
#         }
#     )
#
# with open("annotations.json", "w") as f:
#     json.dump(data, f)
