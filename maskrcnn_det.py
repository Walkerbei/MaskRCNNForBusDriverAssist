#----------------------------------------------------------------------------------------------#
#                                    Rebuild the model                                         #
#                                    JJC | KIKII                                               #
#----------------------------------------------------------------------------------------------#
#-> Update log: 20220509 main model added    #
#----------------------------------------------------------------------------------------------#


import os
import numpy as np 
import torch 
import PIL 
import json
import base64
import math
import requests
import datetime
import time

import torchvision
from torchvision import transforms, datasets, models

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN 
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils


num_classes = 8

label_dict = {'책/문서' : 1, '벨트':2, '안전벨트':3, '음식물':4, '휴대폰':5, '음료':6, '승객':7}

#############################################################################
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask  = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model
#######################################data process##################################################################

root = os.getcwd()
print(root)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = get_model_instance_segmentation(num_classes)
model.to(device)
print(model)



##############main function  ##############
if __name__ == "__main__":
    print(get_model_instance_segmentation(num_classes))
    

