
# data config
class dcfg: 
    input_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/train'
    batch_size = 512 # batch size
    is_gpu_used = True # use GPU or not
    num_workers = 4 # how many workers for loading data
    is_memory_pinned= True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#model config
class mcfg: 
    logger_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/model'
    today = str(date.today())

    is_continued = True
    is_new_version = True
    continued_folder_name = 'continued'
    
    if is_continued:
        folder_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/model/res18/Jun-bsize512-second-source-data-v1'
        model_type = folder_path.split('/')[-2]
        version = folder_path.split('/')[-1]+ continued_folder_name if is_new_version else folder_path.split('/')[-1]        

        ckpt_dir_path = os.path.join(folder_path, 'checkpoints')
        ckpt_names = [file for file in os.listdir(ckpt_dir_path) if file.endswith('ckpt')]
        ckpt_path = os.path.join(ckpt_dir_path, ckpt_names[-1])
    
    else:
        model_type = 'res18'
        version_num = '1235.v5'
        version = f'{today}.{version_num}'
    
    model_folder_path = os.path.join(logger_path, model_type, version)

    is_pretrained = True
    is_customized = False
    raw_model = models.resnet18(pretrained=is_pretrained)
    pred_size = len(word_classes)
    log_every_n_steps = 100
    save_every_n_epoch = 20
    save_top_k_models = 2
    max_epochs = 150
    gpus = 1
    lr = 1e-3 # 3e-4
    
    is_apex_used = True
    amp_level = 'O1'
    precision = 16
    other_setting = 'please write other setting here'

# optimizer configure
class ocfg:
    optim_name = 'Adam'
    lr = 1e-3
    has_differ_lr = True    
    lr_group = [lr/100, lr/10, lr] if has_differ_lr else [lr]
    weight_decay = 0
    momentum = 0.9 if optim_name == 'SGD' else 0 
    
    has_scheduler = False
    schdlr_name = 'OneCycleLR' if has_scheduler else None
    total_steps = (mcfg.max_epochs)*(len(train_image_paths)//dcfg.batch_size+1) if has_scheduler and schdlr_name == 'OneCycleLR' else None
    max_lr = [lr*5 for lr in lr_group] if has_scheduler and schdlr_name == 'OneCycleLR' else None
    other_params = "..."

def save_config(folder_path='/content/', model=None):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
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
                    'max_lr': ocfg.max_lr,
                    'other_params': ocfg.other_params
                }
            }
        },
        'Apex': {
            'is_apex_used': mcfg.is_apex_used,
            'amp_level': mcfg.amp_level,
            'precision': mcfg.precision
        },
        'max_epochs': mcfg.max_epochs,
        'other_settings': mcfg.other_setting        
    }

    with open(os.path.join(folder_path, 'config.json'), 'w') as out_file:
        json.dump(output_dict, out_file, ensure_ascii=False, indent=4)