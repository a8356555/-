# dataset.py
import re
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch as dalitorch
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from .preprocess import transform_func, second_source_transform_func, dali_custom_func, noisy_student_transform_func
from .utils import ImageReader, NoisyStudentDataHandler
from .config import DCFG, MCFG

# TODO: decouple gray, add dali augmentation
#
class BasicDataset(Dataset):
    """Parent Class for all Dataset
    Init Arguments:
        inp_dict: dict, with key value pair {'label': labels, 'image', images, ...}
        transform: function
    """
    def __init__(self, inp_dict, transform=None):
        self.labels = inp_dict['label']
        self.images = inp_dict['image']
        self.image_paths = inp_dict['image_paths']
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

class CustomDataset(BasicDataset):
    pass


class YuShanDataset(Dataset):
    def __init__(self, inp_dict, transform=None): 
        super().__init__(inp_dict, transform)    
        self.labels = inp_dict['int_label']        


class YushanDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, valid_dataset):
        super().__init__()
        self.train = train_dataset
        self.valid = valid_dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=DCFG.batch_size, num_workers=DCFG.num_workers, pin_memory=DCFG.is_memory_pinned, shuffle=DCFG.is_shuffle)
        
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=DCFG.batch_size, num_workers=DCFG.num_workers, pin_memory=DCFG.is_memory_pinned)        



# ------------------
# DALI
# ------------------
class BasicCustomPipeline(Pipeline):
    def __init__(self, 
            image_paths, 
            labels, 
            batch_size=DCFG.batch_size, 
            num_workers=DCFG.num_workers, 
            phase='train', 
            device_id=0
        ):
        super(TrainPipeline, self).__init__(batch_size, num_workers, device_id, exec_async=False, exec_pipelined=False, seed=42)        
        random_shuffle = True if phase == 'train' else False
        self.input = ops.readers.File(files=list(image_paths), labels=list(labels), random_shuffle=random_shuffle, name="Reader")
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.dali_device = 'gpu'
        self.resize = ops.Resize(device=self.dali_device)
        self.crop = ops.Crop(device=self.dali_device, crop=[224.0, 224.0], dtype=types.FLOAT)
        self.transpose = ops.Transpose(device=self.dali_device, perm=[2, 0, 1])
        self.phase = phase

    def define_graph(self):
        pass


class AddRotatePipeline(BasicCustomPipeline):
    def __init__(self, 
            image_paths, 
            labels, 
            batch_size=DCFG.batch_size, 
            num_workers=DCFG.num_workers, 
            phase='train', 
            device_id=0
        ):        
        super().__init__(batch_size, num_workers, device_id, exec_async=False, exec_pipelined=False, seed=42)
        self.rotate = ops.Rotate(device=self.dali_device)  

    def define_graph(self):
        angle = fn.random.uniform(values=[0]*8 + [90.0, -90.0]) # 20% change rotate
        self.jpegs, self.labels = self.input() # (name='r')
        output = self.decode(self.jpegs)
        output = fn.python_function(output, function=dali_custom_func)
        if 'gray' in MCFG.model_type:
            output = fn.color_space_conversion(output, image_type=types.RGB, output_type=types.GRAY)
        if self.phase == 'train':
            w = fn.random.uniform(range=(224.0, 320.0))
            h = fn.random.uniform(range=(224.0, 320.0))        
            output = self.resize(output, resize_x=w, resize_y=h)        
        else:
            output = self.resize(output, size=(248.0, 248.0))
        output = self.crop(output)        
        output = self.rotate(output, angle=angle)
        output = self.transpose(output)
        output = output/255.0
        return (output, self.labels)

class NoisyStudentPipeline(BasicCustomPipeline):
    def __init__(self, 
            image_paths, 
            labels, 
            batch_size=DCFG.batch_size, 
            num_workers=DCFG.num_workers, 
            phase='train', 
            device_id=0
        ):        
        super().__init__(batch_size, num_workers, device_id, exec_async=False, exec_pipelined=False, seed=42)
        self.rotate = ops.Rotate(device=self.dali_device)  
        self.gaussian_blur = ops.
        self.jitter
    def define_graph(self):
        angle = fn.random.uniform(values=[0]*8 + [90.0, -90.0]) # 20% change rotate
        self.jpegs, self.labels = self.input() # (name='r')
        output = self.decode(self.jpegs)
        output = fn.python_function(output, function=dali_custom_func)
        if 'gary' in MCFG.model_type:
            output = fn.color_space_conversion(output, image_type=types.RGB, output_type=types.GRAY)
        if self.phase == 'train':
            w = fn.random.uniform(range=(224.0, 320.0))
            h = fn.random.uniform(range=(224.0, 320.0))        
            output = self.resize(output, resize_x=w, resize_y=h)        
        else:
            output = self.resize(output, size=(248.0, 248.0))
        output = self.crop(output)        
        output = self.rotate(output, angle=angle)
        output = self.transpose(output)
        output = output/255.0
        return (output, self.labels)

class daliModule(pl.LightningDataModule):
    def __init__(self, train_pipeline, valid_pipeline):
        super(daliModule, self).__init__()
        self.pip_train = 
        self.pip_train.build()
        self.pip_valid = 
        self.pip_valid.build()
        self.train_loader = DALIClassificationIterator(self.pip_train, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
        self.valid_loader = DALIClassificationIterator(self.pip_valid, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.valid_loader

def get_input_data_and_transform_func(data_type=OCFG.data_type):
    """
    Arguments:
        data_type: str, 'mixed' or 'cleaned' or '2nd' or 'noisy_student'
    """        
    train_images ,valid_images, train_labels, valid_labels = None, None, None, None
    train_image_paths, valid_image_paths, train_int_labels, valid_int_labels = None, None, None, None
    
    transform_func = transform_func

    if data_type == 'noisy_student':
        ( noised_image_paths, noised_int_labels,
        cleaned_image_paths, cleaned_int_labels,
        valid_image_paths, valid_int_labels) = NoisyStudentDataHandler.get_noisy_student_data(student_iter=NS.student_iter)
        
        train_image_paths = noised_image_paths + cleaned_image_paths        
    else:
        if data_type == 'mixed':
            method_name = 'get_paths_and_int_labels'
            kwargs = {'train_type': 'mixed'}
                    
        elif data_type == 'cleaned':
            method_name = 'get_paths_and_int_labels'
            kwargs = {'train_type': 'cleaned'}
            
        elif data_type == '2nd':
            method_name = 'get_second_source_data' 
            kwargs = {}               
            transform_func = second_source_transform_func
        
        train_image_paths, valid_image_paths, train_int_labels, valid_int_labels = getattr(FileHandler, method_name)(**kwargs)

    valid_images = ImageReader.get_image_data_mp(valid_image_paths, target="image") if is_first_time else None
    train_input_dict = {'image': train_images, 'label': train_labels, 'path': train_image_paths, 'int_label': train_int_labels}
    valid_input_dict = {'image': valid_images, 'label': valid_labels, 'path': valid_image_paths, 'int_label': valid_int_labels}
    return train_input_dict, valid_input_dict, transform_func

def get_datasets(train_input_dict, valid_input_dict, transform_func, is_dali_used=OCFG.is_dali_used, data_type=OCFG.data_type):
    
    if is_dali_used:
        train_dataset = Pipeline(train_input_dict['path'], train_input_dict['int_label'])
        valid_dataset = Pipeline(valid_input_dict['path'], valid_input_dict['int_label'], phase='valid')

    if data_type == 'noisy_student':
        if is_dali_used:
            train_dataset = 
            valid_dataset = 
    
    return train_dataset, valid_dataset

def create_datamodule(is_dali_used=OCFG.is_dali_used, data_type=OCFG.data_type):
    train_input_dict, valid_input_dict, transform_func = get_input_data_and_transform_func(data_type)
    train_dataset, valid_dataset = get_datasets(train_input_dict, valid_input_dict, transform_func, is_dali_used=OCFG.is_dali_used, data_type=OCFG.data_type) 
    datamodule = YushanDataModule(train_dataset, valid_dataset)
    return datamodule