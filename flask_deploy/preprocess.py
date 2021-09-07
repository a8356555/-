import math
import cv2
import base64
import hashlib
import numpy as np
import os
import torch

TEST_DATA_FOLDER = './test_data/'

def _calculate_dhdw_half(h, w):
    """Calculate difference of h or w in order to get a square """
    if h > w:
        dh_half = int(0.1*h/2)
        dw_half = int((h+2*dh_half-w)/2)
    else:
        dw_half = int(0.1*w/2)
        dh_half = int((w+2*dw_half-h)/2)
    return dh_half, dw_half

def preprocess(image):
    # 加邊框
    h, w, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dh_half, dw_half = _calculate_dhdw_half(h, w)
    image = cv2.copyMakeBorder(image, dh_half, dh_half, dw_half, dw_half, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (248, 248))[12:236, 12:236]
    tensor = torch.tensor(image, dtype=torch.float)
    return tensor.permute(2, 0, 1).div(255.0).unsqueeze(0)