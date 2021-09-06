Experimenting on 

https://colab.research.google.com/drive/1_9XwlP7vMmk7C5tZ4Bb-PDSoCeevdOo-#scrollTo=UWss9gCGhiiO

Data checking on

https://colab.research.google.com/drive/1qeCqtGTHLGT2Ha2H7GdWVqKBElVUCcHG

# Table of Contents
1. [YuShan AI Competition](#yac)
    1. [Environmental info](#ei)
    2. [Usage Note](#un)
    3. [Target](#ta1)
    4. [Experiment](#ex1)
    5. [TODO](#todo1)
2. [Noisy Label Skills & Papers](#nls)
    1. [Dataset info](#di)
    2. [Target](#ta1)
    3. [Papers](#p)
    4. [Experiment](#ex1)
    5. [TODO](#todo1)
# <a name="yac">1. YuShan AI Competition (Chinese Word Classification)
Test Accuracy 90% so far

## <a name="ei">Environmental info
    Running on Colab:
        Ubuntu 18.04.5 LTS
        Python 3.7.11
        cuda upgrade to 11.0
        cudnn 7.6.5
        
    Python pytorch related version:
        torch==1.9.0+cu102
        torchvision==0.10.0+cu102
        
## <a name="un">Usage Note
#### 1. Please check whether you install the following package,        
    pip install pytorch-lightning efficientnet_pytorch cupy-cuda110
    pip install --upgrade --force-reinstall --no-deps albumentations        
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110     
    pip install pytorch-lightning
    pip install cupy-cuda110 "if not using gpu

#### 2. Please Modify data/model/optimizer config in config.py First
    a. dcfg: data config
    b. mcfg: model config
    c. ocfg: optimizer config

#### 3. CLI usage (not tested)
    python3 main.py
        -s --stage train or predict stage
        [-i --input-image /path/to/your/image/to/be/predicted]
        [-m model_type] 
        [-c /path/to/your/own/checkpoint/file ] 
        [-t target metric used for evaluating the best model]

## <a name="ta1">Target
    a. Be Familiar with Pytorch Lightning 
    
    b. Deploy on flask + gcp using the api provided by the organizer (app on flask should respond in 1 second)
    
    b. Be Familiar with the following training tricks or tools:
        Tools: Apex, Dali, pytorch profiler, HDF5, 
        Tricks: different learning rate, learning rate scheduler, gradient check
        
    c. Try to speed up training using the following tricks or tools:
        Tools: Apex, Dali, 
        (Bottleneck) Data loading: HDF5, LMDB, TFRecord, tmpfs, hyperparameters (batch_size + num_threads), data prefetcher (not worked)
        
    d. Try to fine tune on Resnet / EfficientNet (Due to the demand of both accuracy and inference speed)
    
    e. Try different augmentation using Dali pipeline including custom python numpy function 

## <a name="ex1">Experiment
    a. batch_size + num_thread (The bigger one is not the better one)
        num_thread = 4 or 8 is the fastest (just for this project)

    b. dataloader: lmdb vs hdf5 vs dali vs raw loading
        lmdb: 3s/batch (using the most disk storage and more time to save)
        hdf5: 2s/batch (using many disk storage, and the save time is large when using high compression level)
        dali: 3s/batch 
        raw: 4~5s/batch

    c. read data: single process vs multi-processing
        single process: 3m48s / 1000 imgs
        multi-process: 17s / 1000 imgs
        Q: why images readed but deleted are still fast to read though not in RAM.
        
    d. read image: pil vs cv2
        讀取時PIL比較快，但那是因為PIL只先打開不讀入，若牽扯到之後的操作包含resize，則使用CV2較快

    e. model hyperparameter for training
    
## <a name="todo1">TODO
    a. use multi-thread to speed up dataloader
    
    b. Further improve model performance (test accuracy 90% now)
    
    c. Try to replace more layers with custom ones
    
    d. Try to gather more input data (gather real data or use GAN to generate training images)
    
    e. Try to implement noisy label related skills from papers


# <a name="nls">2. Noisy Label Skills & Papers
## <a name="di">Dataset info
## <a name="ta2">Target
## <a name="p">Papers
## <a name="ex2">Experiment
## <a name="todo2">TODO
