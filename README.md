Experimenting on 

https://colab.research.google.com/drive/1_9XwlP7vMmk7C5tZ4Bb-PDSoCeevdOo-#scrollTo=UWss9gCGhiiO

Data info and error handling on 

https://colab.research.google.com/drive/1qeCqtGTHLGT2Ha2H7GdWVqKBElVUCcHG

# Table of Contents
* [YuShan AI Competition](#yac)
    1. [Environmental info](#ei)
    2. [Usage Note](#un)
    3. [Target](#ta1)
    4. [Experiment](#ex1)
    5. [TODO](#todo1)
# <a name="yac">YuShan AI Competition (Chinese Word Classification)
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

#### 2. Please Modify data/model/optimizer config in config.py First<br>
* dcfg: data config<br>
* mcfg: model config<br>
* ocfg: optimizer config

#### 3. CLI usage (not tested)
    python3 main.py
        -s --stage train or predict stage
        [-i --input-image /path/to/your/image/to/be/predicted]
        [-m model_type] 
        [-c /path/to/your/own/checkpoint/file ] 
        [-t target metric used for evaluating the best model]

## <a name="ta1">Target
1. Be Familiar with Pytorch Lightning.

2. Deploy on flask + gcp using the api provided by the organizer. (App on flask should respond in 1 second)

3. Be Familiar with the following training tricks or tools:<br>
    Tools: Apex, Dali, pytorch profiler, HDF5,<br>
    Tricks: different learning rate, learning rate scheduler, gradient check

4. Try to speed up training using the following tricks or tools:<br>
    Tools: Apex, Dali,<br>
    (Bottleneck) Data loading: HDF5, LMDB, TFRecord, tmpfs, hyperparameters (batch_size + num_threads), data prefetcher (not worked)

5. Try to fine tune on Resnet / EfficientNet. (Due to the demand of both accuracy and inference speed)

6. Try different augmentation using Dali pipeline including custom python numpy function. 

## <a name="ex1">Experiment
1. batch_size + num_thread (The bigger one is not the better one)<br>
    num_thread = 4 or 8 is the fastest (just for this project)

2. dataloader: lmdb vs hdf5 vs dali vs raw loading<br>
    lmdb: 3s/batch (using the most disk storage and more time to save)<br>
    hdf5: 2s/batch (using many disk storage, and the save time is large when using high compression level)<br>
    dali: 3s/batch<br>
    raw: 4~5s/batch<br>

3. read data: single process vs multi-processing<br>
    single process: 3m48s / 1000 imgs<br>
    multi-process: 17s / 1000 imgs<br>
    Q: why images readed but deleted are still fast to read though not in RAM.

4. read image: pil vs cv2<br>
    讀取時 PIL 比較快，但那是因為 PIL 只先打開不讀入，若牽扯到之後的操作包含resize，則使用CV2較快

5. model hyperparameter for training
    
## <a name="todo1">TODO
1. Use multi-thread to speed up dataloader.

2. Further improve model performance. (test accuracy 90% now)

3. Try to replace more layers with custom ones.

4. Try to gather more input data. (gather real data or use GAN to generate training images)

5. Try to implement noisy label related skills from papers.
