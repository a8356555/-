# dataset.py
import re
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

from .preprocess import transform_func, second_source_transform_func
from .utils import ImageReader, NoisyStudentDataHandler
from .config import DCFG, MCFG
from .model import TrainPipeline

class NoisyStudentDataset(Dataset):
    def __len__(self, inp, transform=None):
        self.image_paths = inp['path']
        self.transform = transform
        self.labels = inp['label']
        self.images = inp['image']
    
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

class YuShanDataset(Dataset):
    def __init__(self, inp, transform=None):    
        self.image_paths = inp['path']
        self.transform = transform
        self.int_labels = inp['int_label']
        self.images = inp['image']
             
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.images:
            image = self.images[index]
        else:
            path = self.image_paths[index]
            image = ImageReader.read_image_cv2(path)
        
        label = int(self.int_labels[index])
        
        if self.transform:            
            transformed_image = self.transform(image=image)

        return transformed_image, label



class YushanDataModule(pl.LightningDataModule):
    def __init__(self, CustomDataset, train_input, valid_input, transform=None):
        super().__init__()
        self.transform = transform
 
        assert isinstance(train_input, dict) and isinstance(valid_input, dict)
        self.train_input = train_input
        self.valid_input = valid_input
        self.train = CustomDataset(train_input, transform=transform)        
        self.valid = CustomDataset(valid_input, transform=transform)        

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

def _get_custom_dataset(model_type=MCFG.model_type):    
    return (YuShanDataset if 'noisy_student' in model_type else NoisyStudentDataset)

def create_datamodule(other_settings=MCFG.other_settings):
    original_dataset_kw = ['raw', 'origin']
    second_source_kw = ['second source', '2nd']
    cleaned_data_kw = ['clean', 'original cleaned']
    noisy_student_kw = ['noisy_student', 'noisy student']
    
    setting_keyword = re.search(
        "|".join(original_dataset_kw + second_source_kw + cleaned_data_kw + noisy_student_kw) + "|", 
        other_settings
        ).group(0)
    
    train_images ,valid_images, train_labels, valid_labels = None, None, None, None
    train_image_paths, valid_image_paths, train_int_labels, valid_int_labels = None, None, None, None

    if setting_keyword in original_dataset_kw:
        train_image_paths, train_int_labels, valid_image_paths, valid_int_labels = FileHandler.get_paths_and_int_labels()
    elif setting_keyword in cleaned_data_kw:
        train_image_paths, train_int_labels, valid_image_paths, valid_int_labels = FileHandler.get_paths_and_int_labels(train_type='cleaned')
        transform_func = transform_func
    elif setting_keyword in second_source_kw:
        train_image_paths, train_int_labels, valid_image_paths, valid_int_labels = FileHandler.get_second_source_data()
        transform_func = second_source_transform_func
    elif setting_keyword in noisy_student_kw:
        train_image_paths, train_labels, train_int_labels, valid_image_paths, valid_labels, valid_int_labels = NoisyStudentDataHandler.get_noisy_student_data(student_iter=NS.student_iter)

    valid_images = ImageReader.get_image_data_mp(valid_image_paths, target="image") if is_first_time else None
    train_input = {'image': train_images, 'label': train_labels, 'path': train_image_paths, 'int_label': train_int_labels}
    valid_input = {'image': valid_images, 'label': valid_labels, 'path': valid_image_paths, 'int_label': valid_int_labels}
    
    CustomDataset = _get_custom_dataset()
    datamodule = YushanDataModule(CustomDataset, train_input, valid_input, transform=transform_func)
    return datamodule