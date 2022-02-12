import shutil
import json
import os
import pandas as pd
import tqdm
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

def copyFiles(input_folder, destination_folder, anns):
    '''Copy all files from the different batch directories and paste them in one folder with the proper ID'''

    batches = []

    for batch in os.listdir(input_folder):
        if "batch" in batch:
            batches.append(os.listdir(os.path.join(input_folder, batch)))

    with open(os.path.join(input_folder, anns), "r") as f:
        data = json.loads(f.read())

    imgs = data["images"]

    for idx, batch in enumerate(batches):
        for file in batch:
            for img in imgs:
                if img["file_name"] == f"batch_{idx+1}/{file}":
                    shutil.copy(os.path.join(input_folder,f'batch_{idx+1}/{file}'), destination_folder)
                    shutil.move(os.path.join(destination_folder, file),os.path.join(destination_folder, f"{img['id']}.jpg"))


def create_taco_df(input_folder, anns_name):

    with open(os.path.join(input_folder, anns_name), 'r') as f:
        dataset = json.loads(f.read())

    df = pd.DataFrame.from_dict(dataset["annotations"])
    df.drop(columns=['area', 'iscrowd'], inplace=True)

    categories = pd.DataFrame.from_dict(dataset["categories"])

    df = df.merge(
        categories,
        left_on='category_id',
        right_on='id').drop(columns="id_y").rename(columns={
            "id_x": "ann_id",
            "name": "category"
        })

    images = pd.DataFrame.from_dict(dataset["images"])

    df = df.merge(
        images[["id", "width", "height", "file_name"]],
        left_on="image_id",
        right_on="id").drop(columns="id")

    return df

def cropping(df, input_folder, destination_folder):

    l_img_paths = []
    l_mask_paths = []

    for i, row in tqdm(df.iterrows()):

        # Skips rows with more than one segmentation polygon
        if len(row["segmentation"]) == 1:

            bbox = row['bbox']

            # Opens the image
            im = Image.open(os.path.join(input_folder, row['file_name']))

            # Reshapes the segmentation polygon
            points = np.array(row['segmentation']).reshape(
            (-1, 2)).astype(np.int32)

            # Creates the mask
            mask = cv2.fillPoly(np.asarray(im) * 0, [points], (255, 255, 255))
            mask = Image.fromarray(mask)

            # Crops the image/mask in a square with 10% padding
            delta = max((bbox[0] + bbox[2]), (bbox[1] + bbox[3]))
            im = im.crop((bbox[0] - delta * 0.1, bbox[1] - delta * 0.1,
                      delta * 1.1, delta * 1.1))
            mask = mask.crop((bbox[0] - delta * 0.1, bbox[1] - delta * 0.1,
                          delta * 1.1, delta * 1.1))

            # Resizes the image/mask
            cropped_resized_image = im.resize((128, 128))
            cropped_resized_mask = mask.resize((128, 128))

            # Saves the image/mask
            cropped_resized_image.save(f'{destination_folder}/Cropped_image/{i}.jpeg')
            cropped_resized_mask.save(f'{destination_folder}/Cropped_mask/{i}.jpg')

        l_img_paths.append(f'{destination_folder}/Cropped_image/{i}.jpeg')
        l_mask_paths.append(f'{destination_folder}/Cropped_mask/{i}.jpg')

    cropped = pd.DataFrame({
        "cropped_image": l_img_paths,
        "cropped_mask": l_mask_paths
    })

    cropped["ann_id"] = cropped["cropped_image"].apply(
        lambda x: int(Path(x).stem))

    return cropped
