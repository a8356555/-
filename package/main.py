# running on colab
from google.colab import drive
drive.mount('/content/gdrive')

import os
import random
import numpy as np
import torch

from train import train
from predict import predict

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    print('seeded')

def make_parser():
    parser = ArgumentParser(
        description="Used if u want to predict on specific test image, if u want to train model please modify config.py")
    parser.add_argument(
        '--stage', '-s', type=str, default='train',
        help='train or predict')
    parser.add_argument(
        '--input-path', '-i', type=str,
        help='/path/to/ur/image/or/image/folder')
    parser.add_argument(
        '--model-class', '-m', type=str, default='EfficientClassifier', required=True,
        help='')

    return parser

if __name__ == '__main__':
    seed_torch()
    parser = make_parser()
    args = parser.parse_args()

    if args.stage == "train":        
        model, trainer, data_module = train()
    elif args.stage == "predict":
        predict(args)