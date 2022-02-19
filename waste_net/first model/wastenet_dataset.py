import pandas as pd
import torch
import torch.utils.data
from PIL import Image, ExifTags, ImageDraw


def nb_diff_other(str_list):
    list_cls = eval(str_list)
    counter = 0
    for clas in list_cls:
        if clas != 'Other':
            counter += 1
    return counter


class WastenetDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, transforms=None):
        self.transforms = transforms
        self.imgs = pd.read_csv(dataset_name)
        self.imgs = self.imgs[self.imgs.labels.apply(nb_diff_other) > 0]
        self.class_map = {6: 'Glass bottle', 18: 'Meal carton', 14: 'Other carton',
            5: 'Clear plastic bottle', 7: 'Plastic bottle cap', 12: 'Drink can',
            10: 'Food Can', 4: 'Other plastic bottle', 50: 'Pop tab', 11: 'Aerosol',
            23: 'Glass cup', 39: 'Other plastic wrapper', 57: 'Styrofoam piece',
            36: 'Plastic film', 29: 'Other plastic', 16: 'Drink carton', 8: 'Metal bottle cap',
            45: 'Disposable food container', 33: 'Normal paper', 20: 'Paper cup',
            40: 'Single-use carrier bag', 31: 'Tissues', 13: 'Toilet tube', 42: 'Crisp packet',
            27: 'Plastic lid', 28: 'Metal lid', 15: 'Egg carton', 55: 'Plastic straw',
            34: 'Paper bag', 21: 'Disposable plastic cup', 9: 'Broken glass',
            49: 'Plastic utensils', 26: 'Glass jar', 25: 'Food waste', 54: 'Squeezable tube',
            43: 'Spread tub', 53: 'Shoe', 38: 'Garbage bag', 0: 'Aluminium foil',
            37: 'Six pack rings', 22: 'Foam cup', 56: 'Paper straw', 17: 'Corrugated carton',
            58: 'Unlabeled litter', 2: 'Aluminium blister pack', 1: 'Battery',
            51: 'Rope & strings', 59: 'Cigarette', 47: 'Other plastic container',
            41: 'Polypropylene bag', 52: 'Scrap metal', 30: 'Magazine paper', 19: 'Pizza box',
            48: 'Plastic glooves', 32: 'Wrapping paper', 3: 'Carded blister pack',
            46: 'Foam food container', 44: 'Tupperware', 24: 'Other plastic cup'
            }

    def __getitem__(self, idx):
        # load images
        img_path_based = '/home/selas/louis/TACO/data/'
        row = self.imgs.iloc[idx]
        img_path = img_path_based + row['img_path']
        img = Image.open(img_path)

        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

            exif = img._getexif()

            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)

        except:
            pass

        # get bounding box coordinates for each object
        bboxes = row['bbox']
        bboxes = eval(bboxes)
        objs = []
        image_width = row['img_width']
        image_height = row['img_height']
        row_labels = eval(row['labels'])

        counter = 0
        for bbox in bboxes:
            new_dict = {
                'xmin': float(bbox[0]),
                'ymin': float(bbox[1]),
                'xmax': float(bbox[0] + bbox[2]),
                'ymax': float(bbox[1] + bbox[3]),
                'classes': row_labels[counter],
                'classes_text': self.class_map[row_labels[counter]]
            }
            objs.append(new_dict)
            counter += 1

        num_objs = len(objs)

        boxes = []
        for i in range(num_objs):
            xmin = objs[i]['xmin']
            xmax = objs[i]['xmax']
            ymin = objs[i]['ymin']
            ymax = objs[i]['ymax']
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.as_tensor(row_labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def plot_image(self, idx):
        image, label = self[idx]
        imgd = ImageDraw.Draw(image)
        bboxs = label['boxes'].tolist()
        for bbox in bboxs:
            imgd.rectangle(bbox, outline="red", width=5)
        return image


    def __len__(self):
        return len(self.imgs)
