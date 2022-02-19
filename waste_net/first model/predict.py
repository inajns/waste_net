from PIL import Image, ImageDraw
import torch
import numpy as np
import joblib
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


model = get_instance_segmentation_model(61)

#load weight
device = torch.device('cuda:0')
weights = joblib.load('model_dect_epoch5new.pkl')
model.to(device)
model.load_state_dict(weights)

# predict.py
#image_path = 'TACO/data/img_all/953.jpg'
image_path = '322.jpg'
img = Image.open(image_path)
model.eval()
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

with torch.no_grad():
    X = np.asarray(img) / 255
    X = torch.Tensor(X).transpose(0, -1)
    prediction = model([X.to(device)])


#Draw on Image
for box, score in zip(prediction[0]["boxes"], prediction[0]["scores"]):
    if score > 0.5:
        #        img2 = Image.fromarray(np.uint8(255*img.transpose(0).detach().cpu().numpy()))
        im = ImageDraw.Draw(img)
        im.rectangle([(box[1], box[0]), (box[3], box[2])],
                     outline="blue",
                     width=5)


#get All bbox
bbox_list = []
for box, score in zip(prediction[0]["boxes"], prediction[0]["scores"]):
    if score > 0.01:
        bbox_image = [(box[1], box[0]), (box[3], box[2])]
        bbox_list.append(bbox_image)
