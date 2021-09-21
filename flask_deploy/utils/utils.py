import os 
import cv2
import numpy as np

def _get_word_classes_dict(training_data_dict_path="/home/luhsuanwen/project/flask_deploy/training data dict.txt"):
    assert os.path.exists(training_data_dict_path), 'file does not exists'

    with open(training_data_dict_path, 'r') as file:
        word_classes = [word.rstrip() for word in file.readlines()]
    print(f'no of origin labels: {len(word_classes)},\nno of unique labels: {np.unique(word_classes).shape[0]}')
    word_classes.append('isnull')
    return word_classes

def save_image(image, ts):
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




word_classes = _get_word_classes_dict()