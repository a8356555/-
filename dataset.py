# dataset.py
from torch.utils.data import Dataset, DataLoader, random_split

# !pip install pytorch-lightning
import pytorch_lightning as pl
# !pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

from .preprocess import transform_func, second_source_transform_func
from .utils import ImageReader
from .config import dcfg
from .model import TrainPipeline

class YuShanDataset(Dataset):
    def __init__(self, input, transform=None):    
        self.image_paths = input['path']
        self.transform = transform
        self.int_labels = input['int_label']
        self.images = input['image']   
             
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
    def __init__(self, train_input, valid_input, transform=None):
        super().__init__()
        self.transform = transform
 
        assert isinstance(train_input, dict) and isinstance(valid_input, dict)
        self.train_input = train_input
        self.valid_input = valid_input
        
        self.train = YuShanDataset(train_input, transform=transform)        
        self.valid = YuShanDataset(valid_input, transform=transform)        

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=dcfg.batch_size, num_workers=dcfg.num_workers, pin_memory=dcfg.is_memory_pinned, shuffle=dcfg.is_shuffle)
        
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=dcfg.batch_size, num_workers=dcfg.num_workers, pin_memory=dcfg.is_memory_pinned)        

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


def create_datamodule(args):
    #TODO: adding dali
    if args.source == 'origin':
        train_image_paths, train_int_labels, valid_image_paths, valid_int_labels = FileHandler.get_paths_and_int_labels()
        transform_func = transform_func
        
    elif args.source == 'second':
        train_image_paths, train_int_labels, valid_image_paths, valid_int_labels = FileHandler.get_second_source_data()
        transform_func = second_source_transform_func

    valid_images = ImageReader.get_image_data_mp(valid_image_paths, target="image") if is_first_time else None
    train_input = {'path': train_image_paths, 'int_label': train_int_labels, 'image': None}
    valid_input = {'path': valid_image_paths, 'int_label': valid_int_labels, 'image': valid_images}
    data_module = YushanDataModule(train_input, valid_input, transform=transform_func)
    return datamodule