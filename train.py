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
from model import get_model
from config import dcfg, mcfg, ocfg, save_config
from dataset import create_datamodule

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


def make_parser():
    parser = ArgumentParser(
        description="Train Model on Origin/Second Source Data")
    parser.add_argument(
        '--source', '-s', type=str, default='origin', required=True,
        help='origin or second')
    parser.add_argument(
        '--model', '-m', type=str, default='res18',
        help='custom or res18')
    
    return parser

if __name__ == '__main__':
    seed_torch()
    parser = make_parser()
    args = parser.parse_args()
    model = get_model()    
    save_config(folder_path=mcfg.model_folder_path, model=model)
    
    data_module = create_datamodule(args)
    
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