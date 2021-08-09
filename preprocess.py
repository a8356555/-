import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import albumentations.augmentations.transforms as transforms
import cupy

from .config import DCFG, MCFG

def _get_copyMakeBorder_flag():
    if 'replicate' in DCFG.transform_approach:
        return cv2.BORDER_REPLICATE
    else:
        return cv2.BORDER_WRAP
        
def _custom_opencv(image):
    # 加邊框
    h, w, c = image.shape
    if h > w:
        dh_half = int(0.1*h/2)
        dw_half = int((h+2*dh_half-w)/2)
    else:
        dw_half = int(0.1*w/2)
        dh_half = int((w+2*dw_half-h)/2)
    if 'gray' in MCFG.model_type:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    flag = _get_copyMakeBorder_flag()
    image = cv2.copyMakeBorder(image, dh_half, dh_half, dw_half, dw_half, flag)
    return image

def transform_func(image=None):    
    h = np.random.randint(224, 320)
    w = np.random.randint(224, 320)
    transform = A.Compose([      
                        A.Resize(h, w),  # 變形                                                     
                        A.CenterCrop(224, 224),
                        A.RandomRotate90(p=0.2),
                        ToTensorV2()
                ])
    # image = _custom_opencv(image)    
    return transform(image=image)['image']/255.0

# --------------------------
# Second Source
# --------------------------


def _second_source_custom_opencv(img):
    # 加邊框
    n = np.random.randint(30, 45)
    img = img[n:300-n, n:300-n]    
    p = np.random.uniform(0, 1)
    
    # 70% 在左右使用 wrap 在上下使用白色
    if p > 0.3:
        img = cv2.copyMakeBorder(img, 100, 100, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img = cv2.copyMakeBorder(img, 0, 0, 100, 100, cv2.BORDER_WRAP)
    # 20% 在上下使用 wrap 在左右使用白色
    elif p > 0.1:
        img = cv2.copyMakeBorder(img, 0, 0, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img = cv2.copyMakeBorder(img, 100, 100, 0, 0, cv2.BORDER_WRAP)
    # 10% 全部使用 wrap
    else:
        img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_WRAP)    
    
    y = np.random.randint(40, 160)
    x = np.random.randint(40, 160)
    p = np.random.uniform(0, 1)
    delta_x, delta_y = (400, 0) if p>0.5 else (0, 400)
    r = np.random.randint(150, 215)
    b = np.random.randint(10, 50)
    g = np.random.randint(10, 50)
    w = np.random.randint(1, 5)
    
    x2 = np.random.randint(260, 420)
    y2 = np.random.randint(260, 420)
    p2 = np.random.uniform(0, 1)
    delta_x2, delta_y2 = (-400, 0) if p>0.5 else (0, -400)

    # 篩出有字的位置加上隨機噪音
    word_cond = img<100
    word_shape = img[word_cond].shape    
    img[word_cond] = img[word_cond] + np.random.normal(0, 1, size=word_shape)

    # 篩出背景加上加上噪音
    bg_cond = img>200
    bg_shape = img[bg_cond].shape
    img[bg_cond] = img[bg_cond] - np.random.randint(low=0, high=50, size=bg_shape)
    
    # 加上擬原始數據線條
    if p>0.4:
        img = cv2.line(img, (x, y), (x+delta_x, y+delta_y), (r, g, b), w)
    if p2>0.4:
        img = cv2.line(img, (x2, y2), (x2+delta_x2, y2+delta_y2), (r, g, b), w)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def second_source_transform_func(image=None):    
    h = np.random.randint(224, 320)
    w = np.random.randint(224, 320)
    transform = A.Compose([      
                        A.Resize(h, w, p=0.7),  # 變形                                                     
                        A.CenterCrop(224, 224),
                        transforms.Blur(blur_limit=5, p=0.7),
                        A.RandomRotate90(p=0.1),
                        ToTensorV2()
                ])
    image = _second_source_custom_opencv(image)    
    return transform(image=image)


# -------------------------
# DALI
# -------------------------
def _copymakeborder_wrap(h, w, dh_half, dw_half, bg, image):        
    # top and bottom
    if h < dh_half:
        top_not_filled = dh_half%h
        bottom_not_filled = -top_not_filled or None # avoid 0 index causes error
        copy_times = int(dh_half/h)
        bg[top_not_filled:dh_half, dw_half:dw_half+w] = bg[-dh_half:bottom_not_filled, dw_half:dw_half+w] = cupy.vstack((image for _ in range(copy_times)))
        if top_not_filled:
            bg[0:top_not_filled, dw_half:dw_half+w] = image[bottom_not_filled:, :]
            bg[bottom_not_filled:, dw_half:dw_half+w] = image[0:top_not_filled, :]
    else:
        bg[0:dh_half, dw_half:dw_half+w] = image[-dh_half:, :] # top
        bg[-dh_half:, dw_half:dw_half+w] = image[0:dh_half, :] # bottom

    # left and right
    roi = bg[:, dw_half:dw_half+w]
    if w < dw_half:
        left_not_filled = dw_half%w 
        right_not_filled = -left_not_filled or None # avoid 0 error
        copy_times = int(dw_half/w)
        bg[:, left_not_filled:dw_half] = bg[:, -dw_half:right_not_filled] = cupy.hstack((roi for _ in range(copy_times)))
        if left_not_filled:
            bg[:, 0:left_not_filled] = roi[:, right_not_filled:]
            bg[:, right_not_filled:] = roi[:, 0:left_not_filled]
    else:
        bg[:, 0:dw_half] = roi[:, -dw_half:]
        bg[:, -dw_half:,:] = roi[:, 0:dw_half] 
    return bg

def _copymakeborder_replicate(h, w, dh_half, dw_half, bg, image):
    bg[0:dh_half, 0:dw_half] = image[0,0]
    bg[0:dh_half, dw_half+w:] = image[0,-1]
    bg[dh_half+h:, 0:dw_half] = image[-1,0]
    bg[dh_half+h:, dw_half+w:] = image[-1,-1]

    bg[0:dh_half, dw_half:dw_half+w] = image[0,:]
    bg[dh_half+h:, dw_half:dw_half+w] = image[-1,:]
    bg[dh_half:dh_half+h, 0:dw_half] = cupy.expand_dims(image[:,0], axis=1)
    bg[dh_half:dh_half+h, dw_half+w:] = cupy.expand_dims(image[:,-1], axis=1)
    return bg



def dali_custom_func(image):
    h, w, c = image.shape
    if h > w:
        dh_half = int(0.1*h/2)
        dw_half = int((h+2*dh_half-w)/2)
    else:
        dw_half = int(0.1*w/2)
        dh_half = int((w+2*dw_half-h)/2)
    # 67 302 134 16
    bg = cupy.zeros((h+2*dh_half, w+2*dw_half, 3), dtype=cupy.uint8)
    bg[dh_half:dh_half+h, dw_half:dw_half+w, :] = image

    if 'replicate' in DCFG.transform_approach:
        func = _copymakeborder_replicate
    elif 'wrap' in DCFG.transform_approach:
        func = _copymakeborder_wrap
    return func(h, w, dh_half, dw_half, bg, image)


def preprocess(image):
    # 加邊框
    h, w, c = image.shape
    if h > w:
        dh_half = int(0.1*h/2)
        dw_half = int((h+2*dh_half-w)/2)
    else:
        dw_half = int(0.1*w/2)
        dh_half = int((w+2*dw_half-h)/2)
    image = cv2.copyMakeBorder(image, dh_half, dh_half, dw_half, dw_half, cv2.BORDER_REPLICATE)
    
    transform = A.Compose([      
                    A.Resize(224, 224),  # 變形                                                     
                    ToTensorV2()
    ])
    return transform(image=image)['image']/255.0
