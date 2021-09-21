import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import pytorch_lightning as pl
import os

CLASS_NUM = 801
class EffClassifier(pl.LightningModule):
    """Parent Class for all lightning modules"""
    def __init__(self, raw_model=None):
        super().__init__()        
        self.model = raw_model
        num_input_fts = self.model._fc.in_features
        self.model._fc = nn.Linear(num_input_fts, CLASS_NUM)    

    def forward(self, x):
        return self.model(x)

def get_best_model(model_name):
    folder = "BEST_MODEL/"
    best_model_ckpt_path = [file for file in os.listdir(folder) if file.endswith(".ckpt")][0]
    best_model_ckpt_path = os.path.join(folder, best_model_ckpt_path)
    
    raw_model = EfficientNet.from_pretrained(model_name)
    model = EffClassifier.load_from_checkpoint(best_model_ckpt_path, map_location='cpu', **{"raw_model": raw_model})
    return model