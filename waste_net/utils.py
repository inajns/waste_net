import shutil
import json
import os
import pandas as pd

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
