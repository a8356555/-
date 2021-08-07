# running on colab
from google.colab import drive
drive.mount('/content/gdrive')

import os
import random
import numpy as np
import torch

from train import train
from test import test
from dataset import get_word_classes

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
        description="YuShan Competition")
    parser.add_argument(
        '--source', '-s', type=str, default='origin', required=True,
        help='origin or second')
    parser.add_argument(
        '--stage', '-s', type=str, default='train',
        help='train or test')
    parser.add_argument(
        '--input-path', '-i', type=str,
        help='test image path')

    return parser

if __name__ == '__main__':
    seed_torch()
    parser = make_parser()
    args = parser.parse_args()

    if args.stage == "train":
        model, trainer, data_module = train(args)
    elif args.stage == "test":
        test(args)