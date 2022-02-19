# Imports
import json
import pandas as pd


# Load annotations
with open('annotations.json') as f:
    annotations = json.load(f)


# Create images list
plastic_cat_id = [
    4, 5, 7, 21, 24, 27, 29, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 55
]
glass_cat_id = [6, 9, 23, 26]
carton_cat_id = [3, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 33, 34, 56]

images = []

for i in range(len(annotations['annotations'])):

    if annotations['annotations'][i]['category_id'] in plastic_cat_id:

        img_id = annotations["annotations"][i]['image_id']
        bbox = annotations['annotations'][i]['bbox']
        label = 'Plastic'
        area = annotations['annotations'][i]['area']

        img_details = []

        img_details.append(img_id)
        img_details.append(bbox)
        img_details.append(label)
        img_details.append(area)

        images.append(img_details)

    elif annotations['annotations'][i]['category_id'] in glass_cat_id:

        img_id = annotations["annotations"][i]['image_id']
        bbox = annotations['annotations'][i]['bbox']
        label = 'Glass'
        area = annotations['annotations'][i]['area']

        img_details = []

        img_details.append(img_id)
        img_details.append(bbox)
        img_details.append(label)
        img_details.append(area)

        images.append(img_details)

    elif annotations['annotations'][i]['category_id'] in carton_cat_id:
        img_id = annotations["annotations"][i]['image_id']
        bbox = annotations['annotations'][i]['bbox']
        label = 'Carton'
        area = annotations['annotations'][i]['area']

        img_details = []

        img_details.append(img_id)
        img_details.append(bbox)
        img_details.append(label)
        img_details.append(area)

        images.append(img_details)

    else:
        img_id = annotations["annotations"][i]['image_id']
        bbox = annotations['annotations'][i]['bbox']
        label = 'Other'
        area = annotations['annotations'][i]['area']

        img_details = []

        img_details.append(img_id)
        img_details.append(bbox)
        img_details.append(label)
        img_details.append(area)

        images.append(img_details)

# Create the final DataFrame
df = pd.DataFrame()


# Get the image_id from each image
img_id_list = []

for i in range(len(annotations['images'])):
    img_id = annotations['images'][i]['id']
    img_id_list.append(img_id)

df['image_id'] = img_id_list
df.set_index('image_id')


# Merge by image_id
d = {}

for i, box, label, area in images:
    if i not in d:
        d[i] = [[], []]
        d[i][0].append(box)
        d[i][1].append(label)
        d[i][2].append(area)


    else:
        d[i][0].append(box)
        d[i][1].append(label)
        d[i][2].append(area)

d_new = {
    'bbox': [d[i][0] for i in d.keys()],
    'labels': [d[i][1] for i in d.keys()],
    'area': [d[i][2] for i in d.keys()]
}

df['bbox'] = d_new['bbox']
df['labels'] = d_new['labels']
df['area'] = d_new['area']


#Get the file_name_path
img_path_list = []

for img_id in range(len(df)):
    img_path =  f'/Users/qpple/code/inajns/waste_net/raw_data/img_all/{img_id}.jpg'
    img_path_list.append(img_path)
df['img_path'] = img_path_list


#Get the url
file_url_list = []

for i in range(len(df)):
    file_url = annotations['images'][i]['flickr_url']
    file_url_list.append(file_url)

df['file_url'] = file_url_list


#Get the widgth and Heigth

img_width_list = []
img_height_list = []

for i in range(len(df)):
    img_width = i['width']
    img_width_list.append(img_width)

for i in range(len(df)):
    img_height = i['height']
    img_height_list.append(img_height)

df['img_width'] = img_width_list
df['img_heigth'] = img_height_list
