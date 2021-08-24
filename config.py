# config.py
from datetime import date
import torch
import os
from pathlib import Path
from .utils import ModelFileHandler, ConfigHandler

"""
Please Jump to Bottom to Modify Config
    1) data config: DCFG
    2) model config: MCFG
    3) optimizer config: OCFG
    4) noisy student config: NS
"""

# --------------------
#  Modifying Config
# --------------------

# data config
class DCFG: 
    """Config for Data"""
    input_path = Path('/content/gdrive/MyDrive/SideProject/YuShanCompetition/train')
    is_gpu_used = True # use GPU or not
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_memory_pinned= True
    
    batch_size = 84 # batch size
    num_workers = 4 # how many workers for loading data
    is_shuffle = True
    data_type = 'raw' # raw, mixed, cleaned, noisy_student, 2nd
    transform_approach = "replicate, " # BORDER_TYPE: replicate | wrap, COLOR: gray|,
    is_dali_used = True
    class_num = 801
    expected_num_per_class = 100

#model config
class MCFG: 
    """Config for Model"""
    # model name / folder name
    model_type = 'effb0'       ### model in torchvision.models | effb[0-7] | effb[0]_noisy_student | custom
    model_class_name = 'DaliEffClassifier'  ### DaliEffClassifier | NoisyStudentDaliEffClassifier
    other_settings = ''         ### dali | gray |
    is_continued_training = False         ### 請修改
    
    # model training setting
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
    monitor = 'val_loss'

    root_model_folder = Path('/content/gdrive/MyDrive/SideProject/YuShanCompetition/model')
    today = str(date.today())
    ckpt_path = None
    version = None
    target_version_folder = None

# optimizer configure
class OCFG:
    """Config for Optimizer"""
    optim_name = 'Adam'         ### 請修改
    lr = 1e-3                   ### 請修改    
    has_differ_lr = False         ### 請修改
    lr_group = [lr/100, lr/10, lr]     ### 請修改
    weight_decay = 0                                           ### 請修改
    momentum = 0.9 if optim_name == 'SGD' else 0                    ### 請修改
    
    has_scheduler = False
    schdlr_name = 'OneCycleLR' if has_scheduler else None
    total_steps = (MCFG.max_epochs)*(DCFG.class_num*DCFG.expected_num_per_class//DCFG.batch_size+1) if has_scheduler else None
    max_lr = [lr*10 for lr in lr_group] if has_scheduler else None

class NS:
    """Config for Noisy Student"""
    student_iter = 1
    temperature = 1 # 2 / 3 / 4 /5 / 6 
    dropout_rate = 0.3
    drop_connect_rate = 0.3
    teacher_softmax_temp = 1

# handle MCFG
MCFG.version, MCFG.ckpt_path, MCFG.model_type = ModelFileHandler.select_target_model_ver_and_ckpt(MCFG.root_model_folder, MCFG.model_type, MCFG.today, MCFG.is_continued_training)
MCFG.target_version_folder = MCFG.root_model_folder / MCFG.model_type / MCFG.version
# show some information
config = ConfigHandler.load_config(MCFG.target_version_folder)
print('\nModel Detail Check:\n - model ckpt path: ', MCFG.ckpt_path, '\n - model folder path: ', MCFG.target_version_folder, '\n - model other_settings: ', MCFG.other_settings)
if config:
    print(' - model other_settings before: ', config['other_settings'])