import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
import re
import multiprocessing
import os
import shutil
import tarfile
import torch

# load training data dict
training_data_dict_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/training data dic.txt'
with open(training_data_dict_path, 'r') as file:
    word_classes = [word.rstrip() for word in file.readlines()]
print(f'no of origin labels: {len(word_classes)},\nno of unique labels: {np.unique(word_classes).shape[0]}')
word_classes.append('isnull')

def word2int_label(label):
    """Transform word classes into integer labels (0~800)"""
    word2int_label_dict = dict(zip(word_classes, range(len(word_classes))))
    return word2int_label_dict[label]

def int_label2word(int_label):
    """Transform integer labels into word classes"""
    int_label2word_dict = dict(zip(range(len(word_classes)), word_classes))
    return int_label2word_dict[int_label]

class ImageReader:
    @classmethod
    def read_image_pil(cls, path):
        img = Image.open(path)
        return img

    @classmethod
    def read_image_cv2(cls, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @classmethod
    def show_image_and_label(cls, img_path, label, is_path_showed=False):
        """
        """
        print(label, int_label2word(label),'\n')        
        if is_path_showed:
            print(img_path, '\n')
            
        image = cls.read_image_cv2(img_path)
        plt.imshow(image)
        plt.show()

    @classmethod
    def read_images(cls, file_paths, way='cv2'):
        data = []
        if way == 'cv2':
            data = [[cls.read_image_cv2(path), re.search('[\u4e00-\u9fa5]{1}', path).group(0)] for path in file_paths]                                   
        else:
            data = [[cls.read_image_pil(path), re.search('[\u4e00-\u9fa5]{1}', path).group(0)] for path in file_paths]
        images, labels = zip(*data)
        return images, labels

    @classmethod
    def read_images_mp(cls, file_paths):
        """Using multi-processing to speed up reading images from paths
        """

        def worker(path, return_list, i):
            """worker function"""
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return_list.append(img)
            if (i+1) % 1000 == 0:
                print(f'{i+1} pictures loaded')

        manager = multiprocessing.Manager()
        return_list = manager.list()
        jobs = []
        print(f'total images: {len(file_paths)}\nstart loading...')
        for i, path in enumerate(file_paths):
            p = multiprocessing.Process(target=worker, args=(path, return_list, i))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        return return_list
 
class FileHandler:
    @classmethod
    def tar_file(cls, file_url):
        start = time.time()
        with tarfile.open(file_url) as file:
            file.extractall(os.path.dirname(file_url),)
            print(f'done! time: {time.time() - start}')

    @classmethod
    def copyfolder(cls, src, dst, symlinks=False, ignore=None):
        """Copy entire folder from src to dst"""
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    @classmethod
    def save_path_and_label_as_txt(cls, txt_path, paths):
        with open(txt_path, 'w') as out_file:
            for path in paths:
                label = re.search("[\u4e00-\u9fa5]{1}", path).group(0)
                label = word2int_label(label)
                out_file.write(f'{path} {label}\n')

    @classmethod
    def read_path_and_label_from_txt(cls, txt_path):
        with open(txt_path) as in_file:
            lines = in_file.readlines()
            paths, labels = zip(*[[line.split(' ')[0], line.split(' ')[1].rstrip()] for line in lines])
        return paths, labels


def _get_ckpt_path(folder_path):
    ckpt_dir_path = os.path.join(folder_path, 'checkpoints')
    assert os.path.exists(ckpt_dir_path), 'there\'s still no ckeckpoint folder, please select another model version '
    print('existing ckpt file: ', [file for file in os.listdir(ckpt_dir_path) if file.endswith('ckpt')])
    epoch_num = input('which one to use? enter the entire name: ')
    ckpt_path = os.path.join(ckpt_dir_path, epoch_num)
    while not os.path.exists(ckpt_path):
        epoch_num = input('the file dosen\'t exist, please enter again: ')
        ckpt_path = os.path.join(ckpt_dir_path, epoch_num)
    return ckpt_path

def _handle_ckpt_path_and_model_version(is_continued, root_model_folder, model_type, is_new_version, today, continued_folder_name):
    """
    3 phase
        1. enter model_type
        2. enter model version (date)
        3. enter model checkpoint
    """
    if is_continued:        
        target_model_type_folder = os.path.join(root_model_folder, model_type)        
        while not os.path.exists(target_model_type_folder):            
            print('existing model type: ', [m for m in os.listdir(root_model_folder) if not m.startswith('.')])
            print('invalid input!')
            model_type = input('please input correct existing model type: (list above) ')
            target_model_type_folder = os.path.join(root_model_folder, model_type)
        
        print('existing model version', [v for v in os.listdir(target_model_type_folder) if not v.startswith('.')])
        version = input('please enter model version: ')         
        target_model_folder = os.path.join(target_model_type_folder, version)
        while not os.path.exists(target_model_folder):
            print('invalid input!')
            version = input('please enter correct model version: ')
            target_model_folder = os.path.join(target_model_type_folder, version)

        version = (version + continued_folder_name) if is_new_version else version
        ckpt_path = _get_ckpt_path(target_model_folder)
    else:
        folder = os.path.join(root_model_folder, model_type)
        if not os.path.exists(folder): os.mkdir(folder)
        existing_ver = os.listdir(folder)
        print('existing version: ', [re.search('v[0-9]+', v).group(0) for v in existing_ver if not v.startswith('.')])
        version_num = input('please enter lastest version number(eg. v1): ') 
        version = f'{today}.{version_num}'
        ckpt_path = None
    return ckpt_path, model_type, version


def save_config(folder_path='/content/', model=None):
    parent_folder = os.path.abspath(os.path.dirname(folder_path))
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)
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
    target_path = os.path.join(folder_path, 'config.json')
    check = input(f'confirm saving {target_path}? (yes/y/n/no)')
    if check in ['y', 'yes']:
        print('start saving')
        
        with open(target_path, 'w') as out_file:
            json.dump(output_dict, out_file, ensure_ascii=False, indent=4)
    else:
        print('stop saving')

def load_config(folder_path='.', file_path=None):
    data = None
    if file_path:
        path = file_path
    elif folder_path:
        path = os.path.join(folder_path, 'config.json')
    if not os.path.exists(path):
        print('This is a new model, still not config file')
    else:
        with open(path, 'r') as in_file:
            data = json.load(in_file)
    return data

def get_train_steps