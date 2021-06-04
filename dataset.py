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

def get_data():
    # load training data dict
    training_data_dict_path = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/training data dic.txt'
    
    with open(training_data_dict_path, 'r') as file:
        word_classes = [word.rstrip() for word in file.readlines()]
    
    print(f'no of origin labels: {len(word_classes)},\nno of unique labels: {np.unique(word_classes).shape[0]}')

    word_classes.append('isnull')

    train_txt = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/train_balanced_images.txt'
    valid_txt = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/valid_balanced_images.txt'
    train_image_paths, train_int_labels = FileHandler.read_path_and_label_from_txt(train_txt)
    valid_image_paths, valid_int_labels = FileHandler.read_path_and_label_from_txt(valid_txt)
    return word_classes, train_image_paths, train_int_labels, valid_image_paths, valid_int_labels


def create_datamodule(args):
    if args.source == 'origin':
        word_classes, train_image_paths, train_int_labels, valid_image_paths, valid_int_labels = get_data()    
        transform = A.Compose([                                                                      
                        A.SmallestMaxSize(225),
                        A.RandomCrop(224, 224),
                        ToTensorV2()
        ])
        train_input = {'path': train_image_paths, 'int_label': train_int_labels}
        valid_input = {'path': valid_image_paths, 'int_label': valid_int_labels}
        
        datamodule = YushanDataModule(train_input, valid_input, transform=transform)
    elif args.source == 'origin':
        "..."

    return datamodule