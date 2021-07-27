# dataset.py
from torch.utils.data import Dataset, DataLoader, random_split
from .utils import ImageReader
from .config import dcfg

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


#也可以用傳統的dataloader
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

def get_origin_data(args):    
    # train_txt = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/train_balanced_images.txt'
    # valid_txt = '/content/gdrive/MyDrive/SideProject/YuShanCompetition/valid_balanced_images.txt'
    # train_image_paths, train_int_labels = FileHandler.read_path_and_label_from_txt(train_txt)
    # valid_image_paths, valid_int_labels = FileHandler.read_path_and_label_from_txt(valid_txt)

    def custom_func(gp_df):
        num = gp_df.shape[0]
        gp_df = pd.concat([gp_df]*(100//num+1))
        return gp_df
        
    df_all = pd.read_csv('/content/gdrive/MyDrive/SideProject/YuShanCompetition/all_data.csv')
    df_valid = df_all.groupby('int_label').sample(6)
    valid_image_paths, valid_int_labels = df_valid['path'].to_numpy(), df_valid['int_label'].to_numpy()

    df_balance = df_all[~df_all['path'].isin(valid_image_paths)].groupby('int_label').apply(custom_func).reset_index(drop=True).groupby('label').sample(100)
    train_image_paths, train_int_labels = df_balance['path'].to_numpy(), df_balance['int_label'].to_numpy()
    return train_image_paths, train_int_labels, valid_image_paths, valid_int_labels

def get_second_source_data():
    df_train = pd.read_csv('/content/gdrive/MyDrive/SideProject/YuShanCompetition/new_data_train.csv')
    df_valid = pd.read_csv('/content/gdrive/MyDrive/SideProject/YuShanCompetition/new_data_valid.csv')
    
    train_image_paths = df_train['path'].to_numpy()
    train_int_labels = df_train['int_label'].to_numpy()
    valid_image_paths = df_valid['path'].to_numpy()
    valid_int_labels = df_valid['int_label'].to_numpy()
    return train_image_paths, train_int_labels, valid_image_paths, valid_int_labels

def create_datamodule(args):
    if args.source == 'origin':
        train_image_paths, train_int_labels, valid_image_paths, valid_int_labels = get_origin_data()
        transform_func = transform_func
        
    elif args.source == 'second':
        train_image_paths, train_int_labels, valid_image_paths, valid_int_labels = get_second_source_data()
        transform_func = second_source_transform_func

    valid_images, _ = ImageReader.read_images_mp(valid_image_paths)
    train_input = {'path': train_image_paths, 'int_label': train_int_labels, 'image': train_images}
    valid_input = {'path': valid_image_paths, 'int_label': valid_int_labels, 'image': valid_images}
    data_module = YushanDataModule(train_input, valid_input, transform=transform_func)

    return datamodule