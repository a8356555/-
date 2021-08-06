# config.py
from datetime import date
import torch
import os
import json
from .utils import save_config, load_config, _handle_ckpt_path_and_model_version

# data config
class dcfg: 
    input_path = Path('/content/gdrive/MyDrive/SideProject/YuShanCompetition/train')
    is_gpu_used = True # use GPU or not
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_memory_pinned= True
    batch_size = 128 # batch size
    num_workers = 4 # how many workers for loading data
    is_shuffle = True
    transform_approach = 'replicate + gaussian noise'
    class_num = 801
    expected_num_per_class = 100

#model config
class mcfg: 
    root_model_folder = Path('/content/gdrive/MyDrive/SideProject/YuShanCompetition/model')
    today = str(date.today())

    model_type = 'effb0'       ### 請修改
    description = 'train directly on original data with no second data source, using dali'### 請修改
    is_continued = True         ### 請修改
    is_new_continued_version = False      ### 請修改
    new_continued_folder_name = 'continued_epoch_30'     ### 請修改
    ckpt_path, model_type, version = _handle_ckpt_path_and_model_version(is_continued, root_model_folder, model_type, is_new_continued_version, today, new_continued_folder_name)
    model_folder_path = root_model_folder / model_type / version

    is_pretrained = True        ### 請修改
    is_customized = False       ### 請修改
    max_epochs = 30           ### 請修改
    save_every_n_epoch = 3      ### 請修改

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
    has_differ_lr = False         ### 請修改
    lr_group = [lr/100, lr/10, lr] if has_differ_lr else lr     ### 請修改
    weight_decay = 0                                           ### 請修改
    momentum = 0.9 if optim == 'SGD' else 0                    ### 請修改
    
    has_scheduler = False
    schdlr_name = 'OneCycleLR' if has_scheduler else None
    total_steps = (mcfg.max_epochs)*(dcfg.class_num*dcfg.expected_num_per_class//dcfg.batch_size+1) if has_scheduler else None
    max_lr = [lr*10 for lr in lr_group] if has_scheduler else None


# show some information
config = load_config(mcfg.model_folder_path)
print('\nModel Detail Check:\n - model ckpt path: ', mcfg.ckpt_path, '\n - model folder path: ', mcfg.model_folder_path, '\n - model description: ', mcfg.description)
if config:
    print(' - model description before: ', config['other_settings'])