import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import albumentations.augmentations.transforms as transforms

from .config import dcfg
# preprocess.py
def _get_copyMakeBorder_flag():
    if 'replicate' in dcfg.transform_approach:
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
        
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if 'gray' in mcfg.model_type else image
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
    image = _custom_opencv(image)    
    return transform(image=image)['image']/255.0



def second_source_custom_opencv(img):
    # 加邊框
    n = np.random.randint(30, 45)
    img = img[n:300-n, n:300-n]    
    p = np.random.uniform(0, 1)
    if p > 0.3:
        img = cv2.copyMakeBorder(img, 100, 100, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img = cv2.copyMakeBorder(img, 0, 0, 100, 100, cv2.BORDER_WRAP)
    elif p > 0.1:
        img = cv2.copyMakeBorder(img, 0, 0, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img = cv2.copyMakeBorder(img, 100, 100, 0, 0, cv2.BORDER_WRAP)
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

    word_cond = img<100
    word_shape = img[word_cond].shape
    img[word_cond] = img[word_cond] + np.random.normal(0, 1, size=word_shape)

    bg_cond = img>200
    bg_shape = img[bg_cond].shape
    img[bg_cond] = img[bg_cond] - np.random.randint(low=0, high=50, size=bg_shape)
    
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
    image = second_source_custom_opencv(image)    
    return transform(image=image)