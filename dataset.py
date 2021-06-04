# dataset.py
from torch.utils.data import Dataset, DataLoader, random_split
from functools import ImageReader
from config import dcfg
class YuShanDataset(Dataset):
    def __init__(self, image_paths, int_labels, transform=None):    
        self.image_paths = image_paths
        self.transform = transform
        self.int_labels = int_labels
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = ImageReader.read_image_cv2(path)
        label = int(self.int_labels[index])
        # label = word2int_label(label)
        if self.transform:            
            transformed = self.transform(image=image)
            transformed_image = transformed["image"]

        return transformed_image, label

#也可以用傳統的dataloader
class YushanDataModule(pl.LightningDataModule):
    def __init__(self, train_input, valid_input, transform=None):
        super().__init__()
        self.transform = transform
 
        assert isinstance(train_input, dict) and isinstance(valid_input, dict)
        self.train_input = train_input
        self.valid_input = valid_input
        
        self.train = YuShanDataset(train_input['path'], train_input['int_label'], transform=transform)        
        self.valid = YuShanDataset(valid_input['path'], valid_input['int_label'], transform=transform)        

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=dcfg.batch_size, num_workers=dcfg.num_workers, pin_memory=dcfg.is_memory_pinned)        
        
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=dcfg.batch_size, num_workers=dcfg.num_workers, pin_memory=dcfg.is_memory_pinned)        