# config.py
from datetime import date
import torch
import os
from pathlib import Path
import json

"""
Please Jump to Bottom to Modify Config
    1) data config: dcfg
    2) model config: mcfg
    3) optimizer config: ocfg
"""



# ---------------------
# config function
# ---------------------
def _handle_not_exist_folder(folder_path):
    is_existing = folder_path.exists()
    print(f'test if {folder_path} exists: {is_existing}, if False then mkdir')
    if not is_existing:
        folder_path.mkdir(parents=True)
        print(f'test again if {folder_path} exists: {folder_path.exists()}')

def _get_ckpt_path(folder_path):
    ckpt_dir_path = folder_path / 'checkpoints'
    assert ckpt_dir_path.exists(), 'there\'s still no ckeckpoint folder, please select another model version '
    print('existing ckpt file: ', [x.name for x in ckpt_dir_path.glob("*.ckpt")])
    epoch_num = input('which one to use? enter the entire name: ')
    ckpt_path = ckpt_dir_path / epoch_num
    while not ckpt_path.exists():
        epoch_num = input('the file dosen\'t exist, please enter again: ')
        ckpt_path = ckpt_dir_path / epoch_num
    return ckpt_path

def _handle_ckpt_path_and_model_version(is_continued, root_model_folder, model_type, is_new_continued_version, today, new_continued_folder_name):
    """
    3 phase
        1. enter model_type
        2. enter model version (date)
        3. enter model checkpoint
    """
    if is_continued:        
        target_model_type_folder = root_model_folder / model_type
        while not target_model_type_folder.exists():
            print('existing model type: ', [x.name for x in root_model_folder.glob("*")])
            print('invalid input!')
            model_type = input('please input correct existing model type: (list above) ')
            target_model_type_folder = root_model_folder / model_type
        
        print('existing model version', [x.name for x in target_model_type_folder.glob("*")])
        version = input('please enter model version: ')         
        target_model_folder = target_model_type_folder / version
        while not target_model_folder.exists():
            print('invalid input!')
            version = input('please enter correct model version: ')
            target_model_folder = target_model_type_folder / version

        version = (version + new_continued_folder_name) if is_new_continued_version else version
        ckpt_path = _get_ckpt_path(target_model_folder)
    else:
        folder = root_model_folder / model_type
        _handle_not_exist_folder(folder)
        print('existing version: ', [x.name for x in folder.glob("v[0-9]*")])
        version_num = input('please enter lastest version number(eg. v1): ') 
        version = f'{today}.{version_num}'
        ckpt_path = None
        _handle_not_exist_folder(folder / version)

    return ckpt_path, model_type, version


def save_config(folder_path=Path('/content'), model=None):  
    import json  
    _handle_not_exist_folder(folder_path)
    
    output_dict = {
        'date': str(mcfg.today,),
        'batch_size': dcfg.batch_size,
        'num_workers': dcfg.num_workers,
        'is_memory_pinned': dcfg.is_memory_pinned,
        'model': {
            'model_type': mcfg.model_type,
            'is_pretrained': mcfg.is_pretrained,
            'is_customized': mcfg.is_customized,
            'model_architecture': str(model).split('\n')
        },

        'optimizer': { 
            'name': ocfg.optim_name,
            'learning rate': {
                'params groups': len(ocfg.lr_group) if ocfg.has_differ_lr else 1,
                'lr': ocfg.lr_group if ocfg.has_differ_lr else ocfg.lr,
            },            
            'optimizer params': {
                'momentum': ocfg.momentum,
                'weight_decay': ocfg.weight_decay
            },

            'scheduler': {
                'has_scheduler': ocfg.has_scheduler,
                'name': ocfg.schdlr_name,
                'scheduler params': {
                    'total_steps': ocfg.total_steps,
                    'max_lr': ocfg.max_lr
                }
            }
        },
        'Apex': {
            'is_apex_used': mcfg.is_apex_used,
            'amp_level': mcfg.amp_level,
            'precision': mcfg.precision
        },
        'max_epochs': mcfg.max_epochs,
        'other_settings': mcfg.description        
    }
    print(json.dumps(output_dict, indent=4))
    target_path = folder_path / 'config.json'
    check = input(f'confirm saving {target_path}? (yes/y/n/no)')
    if check in ['y', 'yes']:
        print('start saving')        
        with open(target_path, 'w') as out_file:
            json.dump(output_dict, out_file, ensure_ascii=False, indent=4)
    else:
        print('stop saving')

def load_config(folder_path='.', file_path=None):
    import json
    data = None    
    path = file_path if file_path else folder_path / 'config.json'
    print(f'config file path: {path}')
    if not path.exists():
        print('This is a new model, still not config file')
    else:
        with open(path, 'r') as in_file:
            data = json.load(in_file)
    return data




# --------------------
#  Modifying Config
# --------------------

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
    momentum = 0.9 if optim_name == 'SGD' else 0                    ### 請修改
    
    has_scheduler = False
    schdlr_name = 'OneCycleLR' if has_scheduler else None
    total_steps = (mcfg.max_epochs)*(dcfg.class_num*dcfg.expected_num_per_class//dcfg.batch_size+1) if has_scheduler else None
    max_lr = [lr*10 for lr in lr_group] if has_scheduler else None


# show some information
config = load_config(mcfg.model_folder_path)
print('\nModel Detail Check:\n - model ckpt path: ', mcfg.ckpt_path, '\n - model folder path: ', mcfg.model_folder_path, '\n - model description: ', mcfg.description)
if config:
    print(' - model description before: ', config['other_settings'])