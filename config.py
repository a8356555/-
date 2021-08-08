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

def save_config(folder_path=Path('/content'), model=None, is_user_input_needed=True):  
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
        'other_settings': mcfg.other_settings        
    }
    print(json.dumps(output_dict, indent=4))
    target_path = folder_path / 'config.json'
    check = input(f'confirm saving {target_path}? (yes/y/n/no)') if is_user_input_needed else 'y'
    if check in ['y', 'yes']:
        print('start saving')        
        with open(target_path, 'w') as out_file:
            json.dump(output_dict, out_file, ensure_ascii=False, indent=4)
    else:
        print('stop saving')

def load_config(version_folder_path='.', file_path=None):
    import json
    data = None    
    path = file_path if file_path else version_folder_path / 'config.json'
    print(f'config file path: {path}')
    if not path.exists():
        print('This is a new model, still not config file')
    else:
        with open(path, 'r') as in_file:
            data = json.load(in_file)
    return data

def _handle_not_exist_folder(folder_path):
    is_existing = folder_path.exists()
    print(f'test if {folder_path} exists: {is_existing}, if False then mkdir')
    if not is_existing:
        folder_path.mkdir(parents=True)
        print(f'test again if {folder_path} exists: {folder_path.exists()}')

def _get_ckpt_path(folder_path):
    ckpt_dir_path = folder_path / 'checkpoints'
    # Check whether ckeckpoint folder exists and print out current exsiting ckpt files
    assert ckpt_dir_path.exists(), 'there\'s still no ckeckpoint folder, please select another model version '
    existing_ckpt_files = [x.name for x in ckpt_dir_path.glob("*.ckpt")]
    print('existing ckpt file: ', existing_ckpt_files)
    
    # Enter the ckpt file to use, if there's more than 1 file with the same epoch num than enter the full name
    epoch_num = input('which one to use? enter the epoch number: ')
    ckpt_name = [x for x in existing_ckpt_files if epoch_num in x]
    while len(ckpt_name) == 0: 
        epoch_num = input('there no matched number,  enter the valid epoch number: ')
        ckpt_name = [x for x in existing_ckpt_files if epoch_num in x]
    
    if len(ckpt_name) > 1:
        ckpt_name = input(f"which one to use? enter the entire ckpt file name: {ckpt_name} ")    
    else: # len == 1
        ckpt_name = ckpt_name[0]
    ckpt_path = ckpt_dir_path / ckpt_name

    while not ckpt_path.exists():
        epoch_num = input('the file dosen\'t exist, please enter again: ')
        ckpt_path = ckpt_dir_path / epoch_num
    return ckpt_path

def _handle_ckpt_path_and_model_version(is_continued, root_model_folder, model_type, today):
    """
    3 phase
        1. enter model_type
        2. enter model version (date)
        3. enter model checkpoint
    """
    import json
    if is_continued:        
        target_model_type_folder = root_model_folder / model_type
        # Loop until model_type input is valid (existing)
        while not target_model_type_folder.exists():
            print('existing model type: ', [x.name for x in root_model_folder.glob("*")])
            print('invalid input!')
            model_type = input('Please input correct existing model type: (list above) ')
            target_model_type_folder = root_model_folder / model_type
        
        # Print out existing version folder name and its setting
        existing_ver_paths = [x for x in folder.glob("*v[0-9]*")]
        config = [load_config(x) for x in existing_ver_paths]
        other_settings = [cfg['other_settings'] if cfg is not None else "No Config" for cfg in config]
        print_dict = {x.name: (des + ", Having below CKPT files~" + 
                                ', '.join([x.name for x in x.glob("**/*.ckpt")])
                                if x.glob("*config.json") 
                                else "There's still no config.json") 
                        for x, des in zip(existing_ver_paths, other_settings)}
        print('existing version: \n', json.dumps(print_dict, sort_keys=True, indent=4))
        
        # Choose a specific version and loop until the version input is valid
        version = input('Please enter model version wanted: ')         
        target_model_folder = target_model_type_folder / version
        while not target_model_folder.exists():
            print('invalid input!')
            version = input('please enter correct model version: ')
            target_model_folder = target_model_type_folder / version
        
        # Decide whether adding a new model version which is the continued version from the existing one
        # If enter y/yes, then input continued folder name postfix
        making_new_continued_version = input("Making new continued version folder?: (y/yes/n/no)")
        if making_new_continued_version in ['y', 'yes']:
            new_continued_folder_name = input("Enter new continued ver folder postfix name (eg. continued)")
            version = (version + new_continued_folder_name)
            print('New continued version folder name: {version}')                        
        ckpt_path = _get_ckpt_path(target_model_folder)
    else:
        # If there's no existing model type then add one
        folder = root_model_folder / model_type
        _handle_not_exist_folder(folder)

        # Print out existing version folder name and its setting
        existing_ver_paths = [x for x in folder.glob("*v[0-9]*")]
        config = [load_config(x) for x in existing_ver_paths]
        other_settings = [cfg['other_settings'] if cfg is not None else "No Config" for cfg in config]
        print_dict = {x.name: (des + ", Having below CKPT files~" + 
                                ', '.join([x.name for x in x.glob("**/*.ckpt")])
                                if x.glob("*config.json") 
                                else "There's still no config.json") 
                        for x, des in zip(existing_ver_paths, other_settings)}
        print('existing version: \n', json.dumps(print_dict, sort_keys=True, indent=4))

        # Enter a new version, if the entered version is existing, then loop again
        version_num = input('please enter version number bigger than existing ones(eg. v1): ') 
        version = f'{today}.{version_num}'
        version_folder = folder / version
        while version_folder.exists():
            version_num = input('please enter version number bigger than existing ones(eg. v1): ') 
            version = f'{today}.{version_num}'
            version_folder = folder / version
        
        ckpt_path = None
        _handle_not_exist_folder(version_folder)

    return ckpt_path, model_type, version

def change_config(
    batch_size=128, 
    max_epochs=30,
    gpus=1,
    num_workers=4,
    transform_approach='replicate',
    other_settings='Write Something', 
    model_type=None,
    version_num=None,
    **kwargs
    ):
    """
    kwargs:
        optim_name = 'Adam'
        lr = 1e-3
        has_differ_lr = False
        lr_group = [lr/100, lr/10, lr] if has_differ_lr else lr     ### 請修改
        weight_decay = 0                                           ### 請修改
        momentum = 0.9
    """
    mcfg.batch_size = batch_size  
    mcfg.max_epochs = max_epochs
    mcfg.gpus = gpus
    dcfg.num_workers=num_workers
    if other_settings != 'Write Something':
        assert isinstance(other_settings, str), 'invalid input data type, need to be string'
        mcfg.other_settings = other_settings            
    if transform_approach != 'replicate':
        dcfg.transform_approach = transform_approach    
    if model_type or version_num:
        if model_type:
            mcfg.model_type = model_type
        if version_num:
            mcfg.version = f"{mcfg.today}.{version_num}"         

        model_folder_path = mcfg.root_model_folder / mcfg.model_type / mcfg.version        
        _handle_not_exist_folder(model_folder_path)
    
    if 'optim_name' in kwargs.keys():
        ocfg.optim_name = kwargs['optim_name']
    if 'lr' in kwargs.keys():
        ocfg.lr = kwargs['lr']
    if 'has_differ_lr' in kwargs.keys():
        ocfg.has_differ_lr = kwargs['has_differ_lr']
    if 'lr_group' in kwargs.keys():
        ocfg.lr_group = kwargs['lr_group']
    if 'weight_decay' in kwargs.keys():
        ocfg.weight_decay = kwargs['weight_decay']
    if 'momentum' in kwargs.keys():
        ocfg.momentum = kwargs['momentum']

    for cfg in [dcfg, mcfg, ocfg]:
        print(cfg)
        for k, v in cfg.__dict__.items():
            print(f"    {k}:  {v}")




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
    transform_approach = 'replicate'
    class_num = 801
    expected_num_per_class = 100

#model config
class mcfg: 
    # model name / folder name
    model_type = 'effb0'       ### 請修改 eg res18 / effb0 / gray effb0
    other_settings = 'train directly on original data with no second data source, using dali'### 請修改
    is_continued = False         ### 請修改
    
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
    
    root_model_folder = Path('/content/gdrive/MyDrive/SideProject/YuShanCompetition/model')
    today = str(date.today())
    ckpt_path, model_type, version = _handle_ckpt_path_and_model_version(is_continued, root_model_folder, model_type, today)
    model_folder_path = root_model_folder / model_type / version

# optimizer configure
class ocfg:
    optim_name = 'Adam'         ### 請修改
    lr = 1e-3                   ### 請修改    
    has_differ_lr = False         ### 請修改
    lr_group = [lr/100, lr/10, lr]     ### 請修改
    weight_decay = 0                                           ### 請修改
    momentum = 0.9 if optim_name == 'SGD' else 0                    ### 請修改
    
    has_scheduler = False
    schdlr_name = 'OneCycleLR' if has_scheduler else None
    total_steps = (mcfg.max_epochs)*(dcfg.class_num*dcfg.expected_num_per_class//dcfg.batch_size+1) if has_scheduler else None
    max_lr = [lr*10 for lr in lr_group] if has_scheduler else None


# show some information
config = load_config(mcfg.model_folder_path)
print('\nModel Detail Check:\n - model ckpt path: ', mcfg.ckpt_path, '\n - model folder path: ', mcfg.model_folder_path, '\n - model other_settings: ', mcfg.other_settings)
if config:
    print(' - model other_settings before: ', config['other_settings'])