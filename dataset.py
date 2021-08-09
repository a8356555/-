# dataset.py
import re
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

from .preprocess import transform_func, second_source_transform_func
from .utils import ImageReader, NoisyStudentDataHandler
from .config import DCFG, MCFG
from .model import TrainPipeline

class BasicDataset(Dataset):
    """Parent Class for all Dataset
    Init Arguments:
        inp: dict, with key value pair {'label': labels, 'image', images, ...}
        transform: function
    """
    def __init__(self, inp, transform=None):
        self.labels = inp['label']
        self.images = inp['image']
        self.image_paths = inp['image_paths']
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, index):
        if self.images:
            image = self.images[index]
        else:
            path = self.image_paths[index]
            image = ImageReader.read_image_cv2(path)
        
        label = int(self.labels[index])
        
        if self.transform:            
            transformed_image = self.transform(image=image)

        return transformed_image, label

class NoisyStudentDataset(BasicDataset):
    def __init__(self, inp, transform=None):
        super().__init__(inp, transform)            


class YuShanDataset(Dataset):
    def __init__(self, inp, transform=None): 
        super().__init__(inp, transform)    
        self.labels = inp['int_label']        


class YushanDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, valid_dataset):
        super().__init__()
        self.train = train_dataset
        self.valid = valid_dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=DCFG.batch_size, num_workers=DCFG.num_workers, pin_memory=DCFG.is_memory_pinned, shuffle=DCFG.is_shuffle)
        
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=DCFG.batch_size, num_workers=DCFG.num_workers, pin_memory=DCFG.is_memory_pinned)        


class daliModule(pl.LightningDataModule):
    def __init__(self, train_input, valid_input):
        super(daliModule, self).__init__()
        self.pip_train = TrainPipeline(train_input['path'], train_input['int_label'])
        self.pip_train.build()
        self.pip_valid = TrainPipeline(valid_input['path'], valid_input['int_label'], phase='valid')
        self.pip_valid.build()
        self.train_loader = DALIClassificationIterator(self.pip_train, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
        self.valid_loader = DALIClassificationIterator(self.pip_valid, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
    
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.valid_loader

def get_input_data(other_settings=MCFG.other_settings):
    original_data_kw = ['raw', 'origin']
    second_source_kw = ['second source', '2nd']
    cleaned_data_kw = ['clean', 'original cleaned']
    noisy_student_kw = ['noisy_student', 'noisy student']
    
    setting_keyword = re.search(
            "|".join(original_data_kw + second_source_kw + cleaned_data_kw + noisy_student_kw) + "|", 
            other_settings
        ).group(0)
    
    train_images ,valid_images, train_labels, valid_labels = None, None, None, None
    train_image_paths, valid_image_paths, train_int_labels, valid_int_labels = None, None, None, None

    kwargs = {}
    if setting_keyword in original_data_kw:
        method_name = 'get_paths_and_int_labels'
        kwargs = {'train_type': 'all_train'}
        ( train_image_paths, train_int_labels, 
          valid_image_paths, valid_int_labels ) = FileHandler.get_paths_and_int_labels()
    elif setting_keyword in cleaned_data_kw:
        method_name = 'get_paths_and_int_labels'
        kwargs = {'train_type': 'cleaned'}
        
        ( train_image_paths, train_int_labels, 
          valid_image_paths, valid_int_labels ) = FileHandler.get_paths_and_int_labels(train_type='cleaned')
    elif setting_keyword in second_source_kw:
        method_name = 'get_second_source_data'
        
        ( train_image_paths, train_int_labels, 
          valid_image_paths, valid_int_labels ) = FileHandler.get_second_source_data()
        transform_func = second_source_transform_func        
    elif setting_keyword in noisy_student_kw:
        ( noised_image_paths, noised_int_labels, noised_labels,
          cleaned_image_paths, cleaned_int_labels, cleand_labels, 
          valid_image_paths, valid_int_labels, valid_labels ) = NoisyStudentDataHandler.get_noisy_student_data(student_iter=NS.student_iter)

    valid_images = ImageReader.get_image_data_mp(valid_image_paths, target="image") if is_first_time else None
    train_input = {'image': train_images, 'label': train_labels, 'path': train_image_paths, 'int_label': train_int_labels}
    valid_input = {'image': valid_images, 'label': valid_labels, 'path': valid_image_paths, 'int_label': valid_int_labels}
    return 

def get_dataset(model_type=MCFG.model_type):    
    return (YuShanDataset if 'noisy_student' in model_type else NoisyStudentDataset)

def create_datamodule():
    CustomDataset = _get_custom_dataset()
    train_dataset = CustomDataset(train_input, transform=transform)        
    valid_dataset = CustomDataset(valid_input, transform=transform)        
    

    datamodule = YushanDataModule(train_dataset, valid_dataset)
    return datamodule