import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
import re
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
from tensorboard.backend.event_processing import event_accumulator

# please see #TODO
ROOT = "/content/gdrive/MyDrive/SideProject/YushanChineseWordClassification"


class ImageReader:
    """A class of methods involving in image reading and showing"""
    __slots__ = []
    @classmethod
    def read_image_pil(cls, path):
        img = Image.open(path)
        return img

    @classmethod
    def read_image_RGB_cv2(cls, path):
        """Read and change RGB order"""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @classmethod
    def read_and_show_image(cls, path):
        img = cv2.imread(path)
        plt.imshow(img)
        plt.show()
        return img

    @classmethod
    def show_image_and_label(cls, img_path, label, is_path_showed=False):
        print(label,'\n')        
        if is_path_showed:
            print(img_path, '\n')            
        cls.read_and_show_image(img_path)

    @classmethod
    def read_images_and_labels(cls, file_paths, way='cv2'):
        data = []
        if way == 'cv2':
            data = [[cls.read_image_RGB_cv2(path), re.search('[\u4e00-\u9fa5]{1}', path).group(0)] for path in file_paths]                                   
        else:
            data = [[cls.read_image_pil(path), re.search('[\u4e00-\u9fa5]{1}', path).group(0)] for path in file_paths]
        images, labels = zip(*data)
        return images, labels

    @classmethod
    def _mp_for_images(cls, file_paths, target=None):
        def get_worker(path, i, return_list):
            img = cv2.imread(path)
            if target == "shape":
                return_list.append([img.shape, i])
            elif target == "image":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return_list.append([img, i])            
            del img

        def lazy_memory_worker(path, i, return_list):
            cv2.imread(path)
        import multiprocessing as mp
        manager = mp.Manager()
        jobs = []
        length = len(file_paths)        
        if target:
            worker = get_worker 
            return_list = manager.list()
        else:
            worker = lazy_memory_worker
            return_list = None
        print(f'total images: {length}\nstart loading...')
        
        for i, path in enumerate(file_paths):
            p = mp.Process(target=worker, args=(path, i, return_list))
            jobs.append(p)
            p.start()
            if (i+1) % 1000 == 0: 
                print(f'{i} pics loaded')
                for proc in jobs:
                    proc.join()
                jobs = []
        for proc in jobs:
            proc.join()                                
        return [arr[0] for arr in return_list]

    @classmethod
    def lazy_read_image_mp(cls, file_paths):
        """Using multiprocessing to speed up image reading, but just cache into memery without returning images"""            
        cls._mp_for_images(file_paths)

    @classmethod
    def get_image_data_mp(cls, file_paths, target="shape"):
        """using multiprocessing to speed up image reading, and returning images or shape of images"""
        return cls._mp_for_images(file_paths, target)

class FolderHandler:
    """A class of methods involving in folder copying, deleting and creating the desired folder if it is not existing"""
    __slots__ = []
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
    def handle_not_existing_folder(cls, folder):
        if isinstance(folder, str):
            folder = Path(folder)
        is_existing = folder.exists()
        print(f'test if {folder} exists: {is_existing}, if False then mkdir')
        if not is_existing:
            folder.mkdir(parents=True)
            print(f'test again if {folder} exists: {folder.exists()}')

    @classmethod
    def delete_useless_folder(cls, folder):        
        shutil.rmtree(folder)
        print(f"See whether {folder} exists: {os.path.exists(folder)}")

    @classmethod
    def show_folder_structure(cls, root_folder=ROOT+"/model", max_layer=3):
        # TODO
        # prefix components:
        space =  '    '
        branch = '│   '
        # pointers:
        tee =    '├── '
        last =   '└── '

        print(os.path.basename(root_folder))

        def dfs(root_folder, cur_lay_num):
            if cur_lay_num > max_layer:
                return
            
            folder_list = sorted(os.listdir(root_folder))
            list_len = len(folder_list)
            for i, folder in enumerate(folder_list, start=1):            
                tee_or_last = last if i == list_len else tee
                sub_folder = os.path.join(root_folder, folder)
                if not folder.startswith('.') and os.path.isdir(sub_folder):
                    print(branch*(cur_lay_num>1) + space*(cur_lay_num-1) + tee_or_last + folder)
                    dfs(sub_folder, cur_lay_num+1)
        dfs(root_folder, 1)        

class FileHandler:
    """A class of methods involving in file operations such as tarfile, get data from files, make desired files."""
    __slots__ = []
    @classmethod
    def tar_file(cls, file_url):
        import tarfile
        start = time.time()        
        with tarfile.open(file_url) as file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(file, os.path.dirname(file_url))
            print(f'done! time: {time.time() - start}')

    @classmethod
    def save_paths_and_labels_as_txt(cls, txt_path, paths, labels):
        with open(txt_path, 'w') as out_file:
            for path, label in zip(paths, labels):                
                out_file.write(f'{path} {label}\n')

    @classmethod
    def read_path_and_label_from_txt(cls, txt_path):
        with open(txt_path) as in_file:
            lines = in_file.readlines()
            paths, labels = zip(*[[line.split(' ')[0], line.split(' ')[1].rstrip()] for line in lines])
        return paths, labels

    @classmethod
    def get_word_classes_dict(cls, training_data_dict_path=ROOT+"/data_txt/training data dic.txt"):
        assert os.path.exists(training_data_dict_path), 'file does not exists or google drive is not connected'

        with open(training_data_dict_path, 'r') as file:
            word_classes = [word.rstrip() for word in file.readlines()]
        print(f'no of origin labels: {len(word_classes)},\nno of unique labels: {np.unique(word_classes).shape[0]}')
        word_classes.append('isnull')
        return word_classes

    @classmethod
    def load_target_dfs(cls, 
            df_all_path=ROOT+'/all_data.csv', 
            df_revised_path=ROOT+'/df_revised.csv', 
            df_checked_path=ROOT+'/df_checked.csv'):
        df_all = pd.read_csv(df_all_path)
        df_revised = pd.read_csv(df_revised_path)
        df_checked = pd.read_csv(df_checked_path)
        return df_all, df_revised, df_checked

    @classmethod
    def get_paths_and_int_labels(cls, train_type='mixed', train_txt_path=None, valid_txt_path=None):
        """
        Argument:
            train_type: str, 'raw' or 'mixed' or 'cleaned' (default 'mixed')
            train_txt_path: str, "/path/to/your/train/txt" , if this parameter is used then train_type will be ignored
        """        
        if train_txt_path is None:
            if train_type == 'raw':
                train_txt_path = ROOT + '/data_txt/raw_train_balanced_images.txt'
            elif train_type == 'mixed':
                train_txt_path = ROOT + '/data_txt/mixed_train_balanced_images.txt'
            elif train_type == 'cleaned':
                train_txt_path = ROOT + '/data_txt/cleaned_train_balanced_images.txt'

        valid_txt_path = valid_txt_path or ROOT + '/data_txt/valid_balanced_images.txt'
        train_image_paths, train_int_labels = cls.read_path_and_label_from_txt(train_txt_path)
        valid_image_paths, valid_int_labels = cls.read_path_and_label_from_txt(valid_txt_path)
        return train_image_paths, train_int_labels, valid_image_paths, valid_int_labels
    
    @classmethod
    def get_second_source_data(cls):
        df_train = pd.read_csv(ROOT + '/new_data_train.csv')
        df_valid = pd.read_csv(ROOT + '/new_data_valid.csv')
        
        train_image_paths = df_train['path'].to_list()
        train_int_labels = df_train['int_label'].to_list()
        valid_image_paths = df_valid['path'].to_list()
        valid_int_labels = df_valid['int_label'].to_list()
        return train_image_paths, train_int_labels, valid_image_paths, valid_int_labels

    @classmethod
    def _make_raw_df_once(cls):
        root_paths = [ROOT + '/YuShanTrain', ROOT + '/train']
        image_paths = []
        labels = []
        for root_path in root_paths:
            for root, _, files in os.walk(root_path):
                for file in files:
                    re_result = re.search("[\u4e00-\u9fa5]{1}", file)
                    if re_result and 'modified' not in file:
                        labels.append(re_result.group(0))
                        image_paths.append(os.path.join(root, file)) 
            
        df_all = pd.DataFrame(list(zip(image_paths, labels)), columns = ['path', 'label'])
        return df_all
    
    @classmethod
    def _average_copy_grouped_df_func(cls, gp_df):
            num = gp_df.shape[0]
            gp_df = pd.concat([gp_df]*(100//num+1))
            return gp_df
    
    @classmethod
    def _make_raw_train_data_txt_once(cls):
        df_all, df_revised, df_checked = cls.load_target_dfs()
        valid_txt_path = ROOT + '/data_txt/valid_balanced_images.txt'
        valid_image_paths, valid_int_labels = cls.read_path_and_label_from_txt(valid_txt_path)
        df_train_not_null = df_all[~df_all['path'].isin(valid_image_paths)].groupby('label').apply(cls._average_copy_grouped_df_func).reset_index(drop=True).groupby('label').sample(100)
        df_train_null = df_revised[(df_revised['label'] == 'isnull') & 
                            (~df_revised['path'].isin(df_valid_null['path']))]
        df_train = df_train_not_null.append(df_train_null, ignore_index=True)

        raw_train_image_paths, raw_train_int_labels = df_train['path'].to_list(), df_train['int_label'].to_list()
        
        raw_train_txt_path = ROOT + '/data_txt/raw_train_balanced_images.txt'
        cls.save_paths_and_labels_as_txt(raw_train_txt_path, raw_train_image_paths, raw_train_int_labels)

    @classmethod
    def _make_mixed_train_valid_data_txt_once(cls):
        """Call only once to make half raw half cleaned data txt """
        
        df_all, df_revised, df_checked = cls.load_target_dfs()
        
        not_null_cond = (df_revised['is_deleted'] == False) & (df_revised['label'] != 'isnull')
        null_cond = df_revised['label'] == 'isnull'
        df_valid_not_null = df_revised[not_null_cond].groupby('label').sample(12)
        df_valid_null = df_revised[null_cond].sample(50)
        df_valid = df_valid_not_null.append(df_valid_null, ignore_index=True)
        
        valid_image_paths, valid_int_labels = df_valid['path'].to_list(), df_valid['int_label'].to_list()
        
        df_train_not_null_raw = df_all[~df_all['path'].isin(df_checked['path'])]
        df_train_not_null_revised = df_revised[(not_null_cond) & (~df_revised['path'].isin(valid_image_paths))]        
        
        df_train_not_null = df_train_not_null_raw.append(df_train_not_null_revised, ignore_index=True)
        df_train_not_null = df_train_not_null.groupby('int_label').apply(cls._average_copy_grouped_df_func).reset_index(drop=True).groupby('label').sample(100)
        df_train_null = df_revised[(df_revised['label'] == 'isnull') & 
                                      (~df_revised['path'].isin(df_valid_null['path']))]

        df_train = df_train_not_null.append(df_train_null, ignore_index=True)
        
        train_image_paths, train_int_labels = df_train['path'].to_list(), df_train['int_label'].to_list()

        valid_txt_path = ROOT + '/data_txt/valid_balanced_images.txt'
        cls.save_paths_and_labels_as_txt(valid_txt_path, valid_image_paths, valid_int_labels)

        train_txt_path = ROOT + '/data_txt/mixed_train_balanced_images.txt'
        cls.save_paths_and_labels_as_txt(train_txt_path, train_image_paths, train_int_labels)

    @classmethod
    def _make_cleaned_data_once(cls):
        valid_txt_path = ROOT + '/data_txt/valid_balanced_images.txt'
        valid_image_paths, valid_int_labels = cls.read_path_and_label_from_txt(valid_txt_path)
        df_all, df_revised, df_checked = cls.load_target_dfs()

        not_null_cond = (df_revised['is_deleted'] == False) & (df_revised['label'] != 'isnull')
        null_cond = df_revised['label'] == 'isnull'
        df_clean_not_null = df_revised[(~df_revised['path'].isin(valid_image_paths)) & not_null_cond].groupby('label').sample(60, replace=True)
        df_clean_null = df_revised[(~df_revised['path'].isin(valid_image_paths)) & null_cond]
        df_clean = df_clean_null.append(df_clean_not_null, ignore_index=True)
        clean_image_paths, clean_int_labels = df_clean['path'].to_list(), df_clean['int_label'].to_list()
        clean_txt_path = ROOT + '/data_txt/cleaned_train_balanced_images.txt'
        cls.save_paths_and_labels_as_txt(clean_txt_path, clean_image_paths, clean_int_labels)




class ModelFileHandler:
    """A class of methods involving in model checkpoint files, model metrics, model folder"""
    __slots__ = []
    @classmethod
    def delete_useless_model_folder(cls, root_folder=ROOT+"/model"):        
        for model_type in os.listdir(root_folder):            
            model_folder = os.path.join(root_folder, model_type)            
            if os.path.isdir(model_folder):
                for version in os.listdir(model_folder):
                    version_folder = os.path.join(model_folder, version)
                    folder_content = [content for content in os.listdir(version_folder) if not content.startswith("_")]
                    deleter_cond = ('test' in version_folder) or \
                                   (len(folder_content) == 0) or \
                                   ('config.json' not in folder_content and 'checkpoints' not in folder_content)
                    if deleter_cond:
                        FolderHandler.delete_useless_folder(version_folder)
                    
    @classmethod
    def get_best_metrics_record(cls, folder, target_metric="val_loss", record_num=0, is_process_showed=False):
        """Get best metrics from tensorboard files
        Arguments:
            target_metric: str, "val_acc" or "val_loss"
            record_num: int, 0 stands for loading all records into memory, may OOM. 1 stands for 1 record
        """
        if isinstance(folder, str):
            folder = Path(folder)
        assert "acc" in target_metric or "loss" in target_metric, "target_metric should be accuracy or loss"

        desired_metrics = ["train_epoch_acc", "val_epoch_acc", "train_acc_epoch", "val_acc_epoch", "train_loss_epoch", "val_loss_epoch", "epoch"] # 前兩項為舊紀錄之兼容
        def _get_best_target_metric_record(record_list):
            if "acc" in target_metric:
                return sorted(record_list, key=lambda record: record.value)[-1]
            elif "loss" in target_metric:
                return sorted(record_list, key=lambda record: record.value)[0]

        def _get_matched_record(record_list, step):            
            return [record for record in record_list if record.step == step][0]

        trn_val, l_acc = target_metric.split("_")     
        best_target_metrics = []
        matched_metrics = {k:[] for k in desired_metrics}
        for path in folder.glob("**/*tfevent*"):
            ea = event_accumulator.EventAccumulator(str(path), size_guidance={event_accumulator.SCALARS:record_num})
            ea.Reload()
            metrics_keys = ea.scalars.Keys()
            matched_target_metric = [key for key in metrics_keys if re.search("(%s){1}_(%s){1}_(epoch){1}"%(trn_val, l_acc), key) or re.search("(%s){1}_(epoch){1}_(%s){1}"%(trn_val, l_acc), key)]
            matched_target_metric = matched_target_metric[0] if matched_target_metric else None            
            if is_process_showed:
                print("matched_target_metric: ", matched_target_metric, ", metrics_keys: ", metrics_keys)
            if matched_target_metric and ea.scalars.Items(matched_target_metric):
                best_trgt_mtrc = _get_best_target_metric_record(ea.scalars.Items(matched_target_metric))
                if is_process_showed:
                    print("matched_target_metric: ", matched_target_metric, ", best_trgt_mtrc: ", best_trgt_mtrc)
                for de_mtrc in desired_metrics:
                    if de_mtrc in metrics_keys and ea.scalars.Items(de_mtrc):
                        matched_record = _get_matched_record(ea.scalars.Items(de_mtrc), best_trgt_mtrc.step)
                        if is_process_showed:
                            print("de_mtrc: ", de_mtrc, ", matched_record: ", matched_record) 
                        matched_metrics[de_mtrc].append(matched_record)
                best_target_metrics.append(best_trgt_mtrc)
                
        if not best_target_metrics:
            return None, "", -1
        final_best_trgt_mtrc = _get_best_target_metric_record(best_target_metrics)
        best_record = ""        
        for de_mtrc in desired_metrics:
            having_match_metrics = len(matched_metrics[de_mtrc]) > 0
            if having_match_metrics:
                final_mtch_rcrd = _get_matched_record(matched_metrics[de_mtrc], final_best_trgt_mtrc.step)
                best_record += f"{de_mtrc}: {final_mtch_rcrd.value}, "
                if de_mtrc == "epoch":
                    epoch = int(final_mtch_rcrd.value)
        return final_best_trgt_mtrc.value, best_record, epoch


    @classmethod
    def print_existing_model_version_and_info(cls, model_folder):
        """Print out existing model version name and its info"""
        if isinstance(model_folder, str):
            model_folder = Path(model_folder)
        
        print_dict = {}
        for version_folder in model_folder.glob("*v[0-9]*"): 
            config = ConfigHandler.load_config(version_folder)
            other_setting = config['other_settings'] if config else "No Config"
            ckpt_files = ', '.join([ckpt_path.name for ckpt_path in version_folder.glob("**/*.ckpt")])
            _, best_record, _ = cls.get_best_metrics_record(version_folder)
            print_dict[version_folder.name] = (other_setting, ckpt_files, best_record)
        print('Existing Versions: \n', json.dumps(print_dict, sort_keys=True, indent=8))

    @classmethod
    def get_best_model_ckpt(cls, raw_model_type, root_model_folder, target_metric="val_acc"):
        """
        Argument:
            target_metric: str, "val_acc" or "val_loss"
        """
        if isinstance(root_model_folder, str):
            root_model_folder = Path(root_model_folder)
        model_folder = root_model_folder / raw_model_type
        assert model_folder.exists(), f"target model type not exists, please check model type again {[model_type.name for model_type in root_model_folder.glob('*')]}"
        
        version_metrics = []
        for version_folder in model_folder.glob("*v[0-9]*"): 
            ckpt_paths = ', '.join([ckpt_path for ckpt_path in version_folder.glob("**/*.ckpt")])
            target_metric_value, best_record, epoch = cls.get_best_metrics_record(version_folder, target_metric=target_metric)            
            if target_metric_value:
                version_metrics.append([target_metric_value, version_folder, epoch])
        
        _, best_version_folder, best_epoch = sorted(version_metrics, key=lambda elems: elems[0])[-1]
        best_ckpt_path = best_version_folder/f"epoch={best_epoch}.ckpt"
        i = 0
        while 1:
            i += 1
            best_ckpt_path = best_version_folder/f"epoch={best_epoch-i}.ckpt"
            if best_ckpt_path.exists(): break
            best_ckpt_path = best_version_folder/f"epoch={best_epoch+i}.ckpt"
            if best_ckpt_path.exists(): break
        return best_ckpt_path

    @classmethod
    def select_ckpt_file_path(cls, version_folder):
        if isinstance(version_folder, str):
            version_folder = Path(version_folder)
        ckpt_folder = version_folder / 'checkpoints'
        # Check whether ckeckpoint folder exists and print out current exsiting ckpt files
        assert ckpt_folder.exists(), 'there\'s still no ckeckpoint folder, please select another model version '
        existing_ckpt_files = [x.name for x in ckpt_folder.glob("*.ckpt")]
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
        ckpt_file_path = ckpt_folder / ckpt_name

        while not ckpt_file_path.exists():
            epoch_num = input('the file dosen\'t exist, please enter again: ')
            ckpt_file_path = ckpt_folder / epoch_num
        return ckpt_file_path

    @classmethod
    def select_target_model_ver_and_ckpt(cls, root_model_folder, model_type, date, is_existing):
        """If the target model is existing, select target model version folder name and ckpt through model_type and date, else make a new version folder.
        """
        if isinstance(root_model_folder, str):
            root_model_folder = Path(root_model_folder)
        if is_existing:        
            model_folder = root_model_folder / model_type
            # Loop until model_type input is valid (existing)
            while not model_folder.exists():
                print('existing model type: ', [x.name for x in root_model_folder.glob("*")])
                print('invalid input!')
                model_type = input('Please input correct existing model type: (list above) ')
                model_folder = root_model_folder / model_type
            
            cls.print_existing_model_version_and_info(model_folder)
            
            # Choose a specific version and loop until the version input is valid
            version = input('Please enter model version wanted: ')         
            target_model_folder = model_folder / version
            while not target_model_folder.exists():
                print('invalid input!')
                version = input('please enter correct model version: ')
                target_model_folder = model_folder / version
            
            # Decide whether adding a new model version which is the continued version from the existing one
            # If enter y/yes, then input continued folder name postfix
            making_new_continued_version = input("Making new continued version folder?: (y/yes/n/no)")
            if making_new_continued_version in ['y', 'yes']:
                new_continued_folder_name = input("Enter new continued ver folder postfix name (eg. continued)")
                version = (version + new_continued_folder_name)
                print('New continued version folder name: {version}')                        
            ckpt_file_path = cls.select_ckpt_file_path(target_model_folder)
        else:
            # If there's no existing model type then add one
            model_folder = root_model_folder / model_type
            FolderHandler.handle_not_existing_folder(model_folder)
            cls.print_existing_model_version_and_info(model_folder)
            
            # Enter a new version, if the entered version is existing, then loop again
            version_num = input('please enter version number bigger than existing ones(eg. v1): ') 
            version = f'{date}.{version_num}'
            version_folder = model_folder / version            
            while version_folder.exists():
                overwrite = input("Overwrite current folder? (y/yes/n/no): ")
                if overwrite in ['y', 'yes']:
                    break
                else:
                    version_num = input("Enter the same number again, or enter version number bigger than existing ones: ")                                                
                    version = f'{date}.{version_num}'
                    version_folder = model_folder / version
            
            ckpt_file_path = None
            FolderHandler.handle_not_existing_folder(version_folder)

        return version, ckpt_file_path, model_type


class ConfigHandler:
    """A class of methods involving in config operations such as saving config, loading config or changing CFG"""
    __slots__ = []
    @classmethod
    def _make_config(cls, CFGs, model):
        DCFG, MCFG, OCFG, NS = None, None, None, None
        for CFG in CFGs:
            if CFG.__name__ == 'DCFG':  
                DCFG = CFG
            elif CFG.__name__ == 'MCFG':  
                MCFG = CFG
            elif CFG.__name__ == 'OCFG':  
                OCFG = CFG
            elif CFG.__name__ == 'NS':  
                NS = CFG
        
        assert DCFG and MCFG, 'Please at lease pass DCFG and MCFG'
        output_dict = {
            'date': str(MCFG.today,),
            'data_type': DCFG.data_type,
            'batch_size': DCFG.batch_size,
            'num_workers': DCFG.num_workers,
            'is_memory_pinned': DCFG.is_memory_pinned,
            'transform_approach': DCFG.transform_approach,
            'model': {
                'model_type': MCFG.model_type,
                'is_pretrained': MCFG.is_pretrained,
                'is_customized': MCFG.is_customized,
                'model_architecture': str(model).split('\n')
            },

            'is_dali_used': DCFG.is_dali_used,
            'Apex': {
                'is_apex_used': MCFG.is_apex_used,
                'amp_level': MCFG.amp_level,
                'precision': MCFG.precision
            },
            'max_epochs': MCFG.max_epochs,
            'other_settings': MCFG.other_settings        
        }
        
        if OCFG:
            output_dict['optimizer'] = { 
                'name': OCFG.optim_name,
                'learning rate': {
                    'params groups': len(OCFG.lr_group) if OCFG.has_differ_lr else 1,
                    'lr': OCFG.lr_group if OCFG.has_differ_lr else OCFG.lr,
                },            
                'optimizer params': {
                    'momentum': OCFG.momentum,
                    'weight_decay': OCFG.weight_decay
                },

                'scheduler': {
                    'has_scheduler': OCFG.has_scheduler,
                    'name': OCFG.schdlr_name,
                    'scheduler params': {
                        'total_steps': OCFG.total_steps,
                        'max_lr': OCFG.max_lr
                    }
                }
            }
        # handle noisy label config
        if 'noisy_student' in MCFG.model_type and NS:
            output_dict['noisy_student'] = {
                'student_iter': NS.student_iter,
                'temperature': NS.temperature,
                'dropout_rate': NS.dropout_rate,
                'drop_connect_rate': NS.drop_connect_rate,
                'teacher_softmax_temp': NS.teacher_softmax_temp
            }
        return output_dict

    @classmethod
    def load_config(cls, version_folder=Path('.'), file_path=None):
        if isinstance(version_folder, str):
            version_folder = Path(version_folder)
        if isinstance(file_path, str):
            file_path = Path(file_path)
        data = None    
        path = file_path if file_path else version_folder / 'config.json'
        print(f'config file path: {path}')
        if not path.exists():
            print('This is a new model, still not config file')
        else:
            with open(path, 'r') as in_file:
                data = json.load(in_file)
        return data
    
    @classmethod
    def save_config(cls, CFGs, folder=Path('/content'), model=None, is_user_input_needed=True):  
        FolderHandler.handle_not_existing_folder(folder)
        output_dict = cls._make_config(CFGs, model)
        print(json.dumps(output_dict, indent=4))
        target_path = folder / 'config.json'
        check = input(f'confirm saving {target_path}? (yes/y/n/no)') if is_user_input_needed else 'y'
        if check in ['y', 'yes']:
            print('start saving')        
            with open(target_path, 'w') as out_file:
                json.dump(output_dict, out_file, ensure_ascii=False, indent=4)
        else:
            print('stop saving')

    @classmethod
    def change_CFGs(cls, CFGs, **kwargs):
        """
        Arguments:
            CFGs: list, eg. [DCFG, MCFG, OCFG, NS]

            kwargs: use keyword arguments or an unpacked dict
            (DCFG related)
                data_type: str,
                batch_size: int,
                num_workers: int,
                transform_approach: str, eg. 'replicate'
            
            (MCFG related)    
                model_type: str, eg. 'noisy_student'
                version_num: str, eg. 'v1'
                max_epochs: int,
                gpus: int,
                other_settings: str, model description
            
            (OCFG related)    
                optim_name: str, eg. 'Adam'
                lr: float, eg. 1e-3
                has_differ_lr: bool,
                lr_group: list, eg. [1e-5, 1e-4, 1e-3]
                weight_decay: float,
                momentum: float, 
            
            (NS related)
                student_iter: int,
                temperature: int, (default 1) 
                dropout_rate: float,
                drop_connect_rate: float,
                teacher_softmax_temp: int,
        """
        MCFG = None
        for CFG in CFGs:
            if CFG.__name__ == 'MCFG':
                MCFG = CFG        
        for key, value in kwargs.items():
            no_CFGs_matched_with_the_key = True
            for CFG in CFGs:
                if hasattr(CFG, key):
                    setattr(CFG, key, value)                
                    no_CFGs_matched_with_the_key = False
            if no_key_matched_with_CFGs:
                print(f"wrong argument: {key}\tThere's no CFG matched with the desired argument, please modify the argument or pass the matched CFG")
            

        if MCFG:    
            if 'version_num' in kwargs.keys():
                MCFG.version = f"{MCFG.today}.{kwargs['version_num']}"        
            target_version_folder = MCFG.root_model_folder / MCFG.model_type / MCFG.version        
            FolderHandler.handle_not_existing_folder(target_version_folder)
            MCFG.target_version_folder = target_version_folder          

        for CFG in CFGs:
            print(CFG.__module__)
            for k, v in CFG.__dict__.items():
                print(f"    {k}:  {v}")

class NoisyStudentDataHandler:
    """A class of methods handling noisy student architecture data"""
    __slots__ = []
    @classmethod
    def save_pseudo_label_to_txt(cls, pseudo_labels, pseudo_label_txt_path):
        with open(pseudo_label_txt_path, 'w') as out_file:
            for sample_pseudo_label in pseudo_labels:
                out_file.write(' '.join(sample_pseudo_label))

    @classmethod
    def read_pseudo_label_from_txt(cls, pseudo_label_txt_path):
        with open(pseudo_label_txt_path) as in_file:
            lines = in_file.readlines()
            pseudo_labels = [line.strip().split(' ') for line in lines]
        return pseudo_labels

    @classmethod
    def get_noisy_student_data(cls, student_iter=0, noised_txt_path=ROOT+"/data_txt/noised_train_balanced_images.txt"):
        noised_labels, cleaned_labels = None, None

        noised_image_paths, noised_int_labels = FileHandler.read_path_and_label_from_txt(noised_txt_path)
        cleaned_image_paths, cleaned_int_labels, valid_image_paths, valid_int_labels = FileHandler.get_paths_and_int_labels(train_type='cleaned')

        return ( noised_image_paths, noised_int_labels,
                 cleaned_image_paths, cleaned_int_labels,
                 valid_image_paths, valid_int_labels )

    @classmethod
    def _make_noisy_student_training_data_once(cls):
        cleaned_image_paths, cleaned_int_labels, valid_image_paths, valid_int_labels = FileHandler.get_paths_and_int_labels(train_type='cleaned')
        
        df_all, df_revised, df_checked = FileHandler.load_target_dfs()
        df_revised_not_used = df_revised[~(df_revised['path'].isin(cleaned_image_paths + valid_image_paths))] # there are some 
        
        df_noised = df_all[~df_all['path'].isin(df_checked['path'])].groupby('label').sample(100, replace=True)
        noised_image_paths, noised_int_labels = df_noised['path'].to_list(), df_noised['int_label'].to_list()
        noised_image_paths += df_revised_not_used['path'].to_list()
        noised_int_labels += df_revised_not_used['int_label'].to_list()
        noised_txt_path = ROOT + '/data_txt/noised_train_balanced_images.txt'
        FileHandler.save_paths_and_labels_as_txt(noised_txt_path, noised_image_paths, noised_int_labels)




word_classes = FileHandler.get_word_classes_dict()

# data functions
def word2int_label(label):
    """Transform word classes into integer labels (0~800)"""
    word2int_label_dict = dict(zip(word_classes, range(len(word_classes))))
    return word2int_label_dict[label]

def int_label2word(int_label):
    """Transform integer labels into word classes"""
    if isinstance(int_label, str):
        int_label = int(int_label)
    int_label2word_dict = dict(zip(range(len(word_classes)), word_classes))
    return int_label2word_dict[int_label]
