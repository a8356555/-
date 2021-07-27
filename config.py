# config.py
from datetime import date
import torch
import os
import json
from .utils import save_config, load_config, _handle_ckpt_path_and_model_version

# data config
class dcfg: 
    input_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/train'
    is_gpu_used = True # use GPU or not
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_memory_pinned= True
    batch_size = 512 # batch size
    num_workers = 4 # how many workers for loading data
    is_shuffle = True

#model config
class mcfg: 
    root_model_folder = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/model'
    today = str(date.today())

    model_type = 'res18'       ### 請修改
    description = 'train on gray scale and directly on original data with no second data source, use dali to speed up'         ### 請修改
    is_continued = False         ### 請修改
    is_new_version = False      ### 請修改
    continued_folder_name = 'dali_test'     ### 請修改
    ckpt_path, model_type, version = _handle_ckpt_path_and_model_version(is_continued, root_model_folder, model_type, is_new_version, today, continued_folder_name)
    model_folder_path = os.path.join(root_model_folder, model_type, version)

    is_pretrained = True        ### 請修改
    is_customized = False       ### 請修改
    max_epochs = 80             ### 請修改
    save_every_n_epoch = 10      ### 請修改    

    pred_size = 801
    log_every_n_steps = 50         
    save_top_k_models = 2    
    gpus = 1
    is_apex_used = True
    amp_level = 'O1'
    precision = 16

# optimizer configure
class ocfg:
    optim_name = 'Adam'         ### 請修改
    lr = 1e-3                   ### 請修改    
    has_differ_lr = True         ### 請修改
    lr_group = [lr/100, lr/10, lr] if has_differ_lr else lr     ### 請修改
    weight_decay = 0                                           ### 請修改
    momentum = 0.9 if optim == 'SGD' else 0                    ### 請修改
    
    has_scheduler = False
    schdlr_name = 'OneCycleLR' if has_scheduler else None
    total_steps = (mcfg.max_epochs)*(len(train_image_paths)//dcfg.batch_size+1) if has_scheduler else None
    max_lr = [lr*10 for lr in lr_group] if has_scheduler else None