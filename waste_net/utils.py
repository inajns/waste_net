import shutil
import json
import os

def copyFiles(input_folder, destination_folder, anns):
    '''Copy all files from the different batch directories and paste them in one folder with the proper ID'''

    batches = []

    for batch in input_folder:
        if "batch" in batch:
            batches.append(os.listdir(os.path.join(input_folder, batch)))

    with open(os.path.join(input_folder, anns), "r") as f:
        data = json.loads(f.read())

    anns = data["annotations"]

    for idx, batch in enumerate(batches):
        for img in batch:
            for ann in anns:
                if ann["file_name"] == f"batch_{idx+1}/{img}":
                    shutil.copy(os.path.join(input_folder,f'batch_{idx+1}/{img}'), destination_folder)
                    shutil.move(os.path.join(input_folder, img),os.path.join(destination_folder, f"{ann['id']}.jpg"))
