import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T


img = Image.open("credit/JPEGImages/image_1.jpg").convert("RGB")
mask = Image.open("credit/SegmentationObjectPNG/image_1.png")

mask = np.array(mask)
# instances are encoded as different colors
obj_ids = np.unique(mask)
# first id is the background, so remove it
obj_ids = obj_ids[1:]
print(obj_ids)
# split the color-encoded mask into a set
# of binary masks
masks = mask == obj_ids[:, None, None]
print(masks)
im = Image.fromarray(mask)
im.save("your_file.jpeg")