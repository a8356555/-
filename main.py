#main.py
!pip install pytorch-lightning
!pip install --upgrade --force-reinstall --no-deps albumentations

from google.colab import drive
drive.mount('/content/gdrive')

import os
import shutil
import random
from datetime import date
import time
import re
import h5py
import json

import pandas as pd
import matplotlib.pyplot as plt

import cv2
import numpy as np

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import albumentations.augmentations.transforms as transforms

# in-project
from functools import DataHandler, ImageReader
from model import YuShanClassifier, DaliYuShanClassifier
from config import dcfg, mcfg, ocfg, save_config

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    print('seeded')

def get_data():
    # load training data dict
    training_data_dict_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/training data dic.txt'
    
    with open(training_data_dict_path, 'r') as file:
        word_classes = [word.rstrip() for word in file.readlines()]
    
    print(f'no of origin labels: {len(word_classes)},\nno of unique labels: {np.unique(word_classes).shape[0]}')

    word_classes.append('isnull')

    train_txt = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/train_balanced_images.txt'
    valid_txt = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/valid_balanced_images.txt'
    train_image_paths, train_int_labels = FileHandler.read_path_and_label_from_txt(train_txt)
    valid_image_paths, valid_int_labels = FileHandler.read_path_and_label_from_txt(valid_txt)
    return word_classes, train_image_paths, train_int_labels, valid_image_paths, valid_int_labels

if __name__ == "__main__":
    word_classes, train_image_paths, train_int_labels, valid_image_paths, valid_int_labels = get_data()

    seed_torch()
    
    transform = A.Compose([                                                                      
                       A.SmallestMaxSize(225),
                       A.RandomCrop(224, 224),
                       ToTensorV2()
    ])
    

    model = YuShanClassifier.load_from_checkpoint(mcfg.ckpt_path) if mcfg.is_continued else YuShanClassifier()
    save_config(folder_path=mcfg.model_folder_path, model=model)

    train_input = {'path': train_image_paths, 'int_label': train_int_labels}
    valid_input = {'path': valid_image_paths, 'int_label': valid_int_labels}

    data_module = YushanDataModule(train_input, valid_input, transform=transform)
    logger = TensorBoardLogger(mcfg.logger_path, name=mcfg.model_type, version=mcfg.version)


    checkpoint_callback = ModelCheckpoint(
        monitor=mcfg.monitor,
        dirpath=os.path.join(mcfg.logger_path, mcfg.model_type, mcfg.version, mcfg.model_ckpt_dirname),
        filename='{epoch}',
        save_top_k = mcfg.save_top_k_models,
        every_n_val_epochs=mcfg.save_every_n_epoch,
    )

    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=mcfg.max_epochs, 
        gpus=mcfg.gpus, 
        precision=mcfg.precision if mcfg.is_apex_used else None,
        amp_level=mcfg.amp_level if mcfg.is_apex_used else None,
        log_every_n_steps=mcfg.log_every_n_steps, 
        flush_logs_every_n_steps=mcfg.log_every_n_steps,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=mcfg.ckpt_path if mcfg.is_continued else None
    )

    trainer.fit(model, data_module)