import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
import re
import os
import numpy as np

# please see #TODO

class ImageReader:
    @classmethod
    def read_image_pil(cls, path):
        img = Image.open(path)
        return img

    @classmethod
    def read_image_cv2(cls, path):
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
        """"""
        print(label,'\n')        
        if is_path_showed:
            print(img_path, '\n')            
        cls.read_and_show_image(img_path)

    @classmethod
    def read_images_and_labels(cls, file_paths, way='cv2'):
        data = []
        if way == 'cv2':
            data = [[cls.read_image_cv2(path), re.search('[\u4e00-\u9fa5]{1}', path).group(0)] for path in file_paths]                                   
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
        cls._mp_for_images(file_paths)

    @classmethod
    def get_image_data_mp(cls, file_paths, target="shape"):
        return cls._mp_for_images(file_paths, target)
 
class FileHandler:
    @classmethod
    def tar_file(cls, file_url):
        import tarfile
        start = time.time()        
        with tarfile.open(file_url) as file:
            file.extractall(os.path.dirname(file_url),)
            print(f'done! time: {time.time() - start}')

    @classmethod
    def copyfolder(cls, src, dst, symlinks=False, ignore=None):
        """Copy entire folder from src to dst"""
        import shutil
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    @classmethod
    def save_paths_and_labels_as_txt(cls, txt_path, paths, labels):
        with open(txt_path, 'w') as out_file:
            for path, label in zip(paths, labels):                
                out_file.write(f'{path} {label}\n')

    @classmethod
    def make_train_valid_data_txt(cls):
        """Call only once to make """
        def custom_func(gp_df):
            num = gp_df.shape[0]
            gp_df = pd.concat([gp_df]*(100//num+1))
            return gp_df

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
        df_train_not_null = df_train_not_null.groupby('int_label').apply(custom_func).reset_index(drop=True).groupby('label').sample(100)
        df_train_null = df_revised[(df_revised['label'] == 'isnull') & 
                                      (~df_revised['path'].isin(df_valid_null['path']))]

        df_train = df_train_not_null.append(df_train_null, ignore_index=True)
        
        train_image_paths, train_int_labels = df_train['path'].to_list(), df_train['int_label'].to_list()

        valid_txt_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/valid_balanced_images.txt'
        cls.save_paths_and_labels_as_txt(valid_txt_path, valid_image_paths, valid_int_labels)

        train_txt_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/train_balanced_images.txt'
        cls.save_paths_and_labels_as_txt(train_txt_path, train_image_paths, train_int_labels)


    @classmethod
    def read_path_and_label_from_txt(cls, txt_path):
        with open(txt_path) as in_file:
            lines = in_file.readlines()
            paths, labels = zip(*[[line.split(' ')[0], line.split(' ')[1].rstrip()] for line in lines])
        return paths, labels


    @classmethod
    def get_raw_df(cls):
        root_paths = ['/content/gdrive/MyDrive/SideProject/YuShanTrain', '/content/gdrive/MyDrive/SideProject/YuShanCompetition/train']
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
    def get_word_classes_dict(cls, path=None):        
        training_data_dict_path = path if path else "/content/gdrive/MyDrive/SideProject/YuShanCompetition/training data dic.txt"
        assert os.path.exists(training_data_dict_path), 'file does not exists or google drive is not connected'

        with open(training_data_dict_path, 'r') as file:
            word_classes = [word.rstrip() for word in file.readlines()]
        print(f'no of origin labels: {len(word_classes)},\nno of unique labels: {np.unique(word_classes).shape[0]}')
        word_classes.append('isnull')
        return word_classes

    @classmethod
    def load_target_dfs(cls, df_all_path=None, df_revised_path=None, df_checked_path=None):
        df_all_path = df_all_path or '/content/gdrive/MyDrive/SideProject/YuShanCompetition/all_data.csv'
        df_revised_path = df_revised_path or '/content/gdrive/MyDrive/SideProject/YuShanCompetition/df_revised.csv'
        df_checked_path = df_checked_path or '/content/gdrive/MyDrive/SideProject/YuShanCompetition/df_checked.csv'
        df_all = pd.read_csv(df_all_path)
        df_revised = pd.read_csv(df_revised_path)
        df_checked = pd.read_csv(df_checked_path)
        return df_all, df_revised, df_checked

    @classmethod
    def get_paths_and_int_labels(cls, train_txt_path=None, valid_txt_path=None):
        train_txt_path = train_txt_path or '/content/gdrive/MyDrive/SideProject/YuShanCompetition/train_balanced_images.txt'
        valid_txt_path = valid_txt_path or '/content/gdrive/MyDrive/SideProject/YuShanCompetition/valid_balanced_images.txt'
        train_image_paths, train_int_labels = cls.read_path_and_label_from_txt(train_txt_path)
        valid_image_paths, valid_int_labels = cls.read_path_and_label_from_txt(valid_txt_path)
        return train_image_paths, train_int_labels, valid_image_paths, valid_int_labels


    @classmethod
    def get_second_source_data():
        df_train = pd.read_csv('/content/gdrive/MyDrive/SideProject/YuShanCompetition/new_data_train.csv')
        df_valid = pd.read_csv('/content/gdrive/MyDrive/SideProject/YuShanCompetition/new_data_valid.csv')
        
        train_image_paths = df_train['path'].to_list()
        train_int_labels = df_train['int_label'].to_list()
        valid_image_paths = df_valid['path'].to_list()
        valid_int_labels = df_valid['int_label'].to_list()
        return train_image_paths, train_int_labels, valid_image_paths, valid_int_labels



    @classmethod
    def make_noisy_student_training_data():
        # TODO
        valid_txt_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/valid_balanced_images.txt'
        valid_image_paths, valid_int_labels = cls.read_path_and_label_from_txt(valid_txt_path)
        df_all, df_revised, df_checked = load_target_dfs()
    
        not_null_cond = (df_revised['is_deleted'] == False) & (df_revised['label'] != 'isnull')
        null_cond = df_revised['label'] == 'isnull'
        df_clean = df_revised[(~df_revised['path'].isin(valid_image_paths)) & not_null_cond].groupby('label').sample(60)
        clean_image_paths, clean_int_labels = df_clean


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