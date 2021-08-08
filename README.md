"""
### Environmental info
    Run in Colab:
        Ubuntu 18.04.5 LTS
        Python 3.7.11
        cuda upgrade to 11.0
        cudnn 7.6.5
    Python Package    
        torch==1.9.0+cu102
        torchvision==0.10.0+cu102

### 0) Please check whether you install the following package,        
        pip install pytorch-lightning efficientnet_pytorch cupy-cuda110
        pip install --upgrade --force-reinstall --no-deps albumentations        
        pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110        
        pip install pytorch-lightning
        pip install cupy-cuda110 "not tested"

### 1) Please Modify data/model/optimizer config in the bottom if config.py First
1. dcfg: data config
2. mcfg: model config
3. ocfg: optimizer config

### 2) Then Run in Terminal
    python3 main.py
        -s --stage train or test
        -i --input-image /path/to/your/image