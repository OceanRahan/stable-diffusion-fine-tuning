import datasets
from PIL import Image as Img
from datasets import Dataset, Features, Image, Value
import os
import IPython.display as display
import pandas as pd
import requests
import PIL
import matplotlib.pyplot as plt
from transformers import pipeline
from pathlib import Path


# resize images into 512x512 target size

def resize_and_crop_images(folder_path, target_size=512):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img = Img.open(file_path)
            width, height = img.size
            if width <= height:
                new_width = target_size
                new_height = int(height * (target_size / width))
            else:
                new_width = int(width * (target_size / height))
                new_height = target_size
            resized_image = img.resize((new_width, new_height))
            left = (new_width - target_size) // 2
            top = (new_height - target_size) // 2
            right = (new_width + target_size) // 2
            bottom = (new_height + target_size) // 2
            cropped_image = resized_image.crop((left, top, right, bottom))
            cropped_image.save("all_images_processed/" + filename)


# utility function for generating caption

def generate_caption(file_path):
    img = Img.open(file_path)
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    result = pipe(img)
    caption = result[0]['generated_text']
    return caption


# generate preliminary captions using image to text model

def generate_image_caption_csv(folder_path):
    captions = []
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            caption = generate_caption(file_path)
            captions.append(caption)
            images.append(file_path)
    df = pd.DataFrame({"image": images,
                       "text": captions})
    df.to_csv("stable_diffusion_dataset.csv", index=False)


# create dataset and upload to huggingface

def create_dataset():
    features = Features({
        'image': Image(decode=True),
        'text': Value(dtype='string'),
    })
    captions = []
    images = []
    df = pd.read_csv("image-caption.csv")
    for _, row in df.iterrows():
        file_path = row["image"]
        img = Img.open(file_path)
        caption = row["text"]
        images.append(Image().encode_example(img))
        captions.append(caption)
    new_df = pd.DataFrame({"image": images,
                           "text": captions})
    ds = Dataset.from_pandas(new_df, features=features)
    ds.push_to_hub(repo_id="oceanrahan/fine-tune-data-stable-diffusion")
    df.to_csv("stable_diffusion_dataset.csv", index=False)


# create_captions("all_images_processed")
# create_dataset()
# resize_and_crop_images("all_images")
# dataset = datasets.load_dataset("lambdalabs/naruto-blip-captions")
# dataset.save_to_disk("C://Users/ocean/Downloads/")
