import os 
import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
import cv2

def _get_word_classes_dict(training_data_dict_path="training data dic.txt"):
    assert os.path.exists(training_data_dict_path), 'file does not exists or google drive is not connected'

    with open(training_data_dict_path, 'r') as file:
        word_classes = [word.rstrip() for word in file.readlines()]
    print(f'no of origin labels: {len(word_classes)},\nno of unique labels: {np.unique(word_classes).shape[0]}')
    word_classes.append('isnull')
    return word_classes

def save_iamge(image, ts):
    folder = "test_data_saved"
    cv2.imwrite(os.path.join(folder, ts+".jpg"), image)

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

def get_best_model():
    folder = "BEST_MODEL/"
    best_model_ckpt_path = [file for file in os.listdir(folder) if file.endswith(".ckpt")][0]
    checkpoint = torch.load(best_model_ckpt_path, map_location=torch.device('cpu'))

    class_num = 801
    model = EfficientNet.from_pretrained(name)
    num_input_fts = model._fc.in_features
    model._fc = nn.Linear(num_input_fts, class_num)
    model.load_state_dict(checkpoint['state_dict'])
    return model


word_classes = _get_word_classes_dict()