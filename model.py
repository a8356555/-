# model.py
import os
import re
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
import torchvision

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch as dalitorch
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from .config import MCFG, DCFG, OCFG, NS
CFGs = [MCFG, DCFG, OCFG, NS]
from .preprocess import dali_custom_func
from .utils import MetricsHandler, ModelFileHandler

MODEL_BACKBONES = ["eff", "res", "custom"]

class BaiscClassifier(pl.LightningModule):
    """Parent Class for all lightning modules"""
    def __init__(self):
        super().__init__()        
        self.model = _get_raw_model()        
    def forward(self, x):
        return self.model(x)

    def process_batch(self, batch):
      x, y = batch
      return x.float(), y.long()

    def cross_entropy_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)

    def training_step(self, train_batch, batch_idx):               
        x, y = self.process_batch(train_batch)
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)        
        _, y_hat = torch.max(logits, dim=1)
        running_corrects = torch.sum(y_hat == y)            
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'running_corrects': running_corrects, 'batch_size': y.shape[0]}

    def training_epoch_end(self, outputs):
        epoch_corrects = sum([x['running_corrects'] for x in outputs])
        dataset_size = sum([x['batch_size'] for x in outputs])
        acc = epoch_corrects/dataset_size
        loss = sum([x['loss'] for x in outputs])/dataset_size
        print(f"current epoch: {self.current_epoch}")
        print(f"- train_epoch_acc: {acc}, train_loss: {loss}")
        self.log('train_epoch_acc', acc)

    def validation_step(self, val_batch, batch_idx):              
        x, y = self.process_batch(val_batch)
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        _, y_hat = torch.max(logits, dim=1)
        running_corrects = torch.sum(y_hat == y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'running_corrects': running_corrects, 'batch_size': y.shape[0]}
    
    def validation_epoch_end(self, outputs):
        epoch_corrects = sum([x['running_corrects'] for x in outputs])
        dataset_size = sum([x['batch_size'] for x in outputs])
        acc = epoch_corrects/dataset_size  
        loss = sum([x['loss'] for x in outputs])/dataset_size
        
        if (self.current_epoch+1) % MCFG.save_every_n_epoch == 0:
            metrics_txt_path = os.path.join(MCFG.target_version_folder, 'metrics.txt')
            MetricsHandler.save_metrics_to_txt(self.current_epoch, loss, acc, metrics_txt_path)
        print(f'- val_epoch_acc: {acc}, val_loss: {loss}')        
        self.log('val_epoch_acc', acc)    

    def configure_optimizers(self):
        if OCFG.has_differ_lr:
            params_group = self._get_params_group()
            print(params_group)

            optimizer = torch.optim.Adam(params_group, weight_decay=OCFG.weight_decay) if OCFG.optim_name == 'Adam' else \
                        torch.optim.SGD(params_group, momentum=OCFG.momentum, weight_decay=OCFG.weight_decay)

        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=OCFG.lr, weight_decay=OCFG.weight_decay) if OCFG.optim_name == 'Adam' else \
                        torch.optim.SGD(self.model.parameters(), lr=OCFG.lr, momentum=OCFG.momentum, weight_decay=OCFG.weight_decay)

        if OCFG.has_scheduler:
            if OCFG.schdlr_name == 'OneCycleLR':
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=OCFG.max_lr, total_steps=OCFG.total_steps)  # 设置学习率下降策略

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            }
        else:
            return optimizer
    
    def _get_params_group(self):
        """
        customizer for model enabling differ learning rate
        """
        pass

class ResNetClassifier(BaiscClassifier):
    def __init__(self):
        super().__init__()
        num_input_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_input_fts, DCFG.class_num)        
    
    def _get_params_group(self):
        fc_params = list(map(id, self.model.fc.parameters()))
        layer4_params = list(map(id, self.model.layer4.parameters()))
        all_params = list(map(id, self.model.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params + layer4_params,
                            self.model.parameters())
        params_group = [
                        {'params': base_params, 'lr': OCFG.lr_group[0]},
                        {'params': self.model.layer4.parameters(), 'lr': OCFG.lr_group[1]},
                        {'params': self.model.fc.parameters(), 'lr': OCFG.lr_group[2]}
        ]                
        return params_group
        
class EfficientClassifier(BaiscClassifier):
    def __init__(self):
        super().__init__()
        num_input_fts = self.model._fc.in_features
        self.model._fc = nn.Linear(num_input_fts, DCFG.class_num)        
    
    def _get_params_group(self):
        fc_params_id = list(map(id, self.model._fc.parameters()))
        tail_params_id = list(map(id, self.model._blocks[-3:].parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id + tail_params_id,
                            self.model.parameters())
        tail_params = filter(lambda p: id(p) in  tail_params_id,
                            self.model.parameters())
        params_group = [
                        {'params': base_params, 'lr': OCFG.lr_group[0]},
                        {'params': tail_params, 'lr': OCFG.lr_group[1]},
                        {'params': self.model._fc.parameters(), 'lr': OCFG.lr_group[2]}
        ]                
        return params_group

class GrayWrapperModel(nn.Module):
    def __init__(self, raw_model):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, padding=1)
        self.model = raw_model
    def forward(self, x):
        x = self.conv1(x)        
        return self.model(x)

class GrayResClassifier(BaiscClassifier):
    def __init__(self):
        super().__init__()
        self.model = GrayWrapperModel(self.model)
        num_input_fts = self.model.model.fc.in_features
        self.model.model.fc = nn.Linear(num_input_fts, DCFG.class_num)        
                    
    def _get_params_group(self):
        conv1_params_id = list(map(id, self.model.conv1.parameters())) 
        fc_params_id = list(map(id, self.model.model.fc.parameters()))
        layer4_params_id = list(map(id, self.model.model.layer4.parameters()))
        # all_params = list(map(id, self.model.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id + layer4_params_id + conv1_params_id,
                            self.model.parameters())
        params_group = [
                        {'params': base_params, 'lr': OCFG.lr_group[0]},
                        {'params': self.model.model.layer4.parameters(), 'lr': OCFG.lr_group[1]},
                        {'params': self.model.model.fc.parameters(), 'lr': OCFG.lr_group[2]},
                        {'params': self.model.conv1.parameters(), 'lr': OCFG.lr_group[2]}
        ]        
        return params_group 

class GrayEffClassifier(BaiscClassifier):
    def __init__(self):
        super().__init__()        
        self.model = GrayWrapperModel(self.model)    
        num_input_fts = self.model.model._fc.in_features
        self.model.model._fc = nn.Linear(num_input_fts, DCFG.class_num)        
                    
    def _get_params_group(self):
        con1_params_id = list(map(id, self.model.conv1.parameters())) 
        fc_params_id = list(map(id, self.model.model._fc.parameters()))
        tail_params_id = list(map(id, self.model.model._blocks[-3:].parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id + tail_params_id + con1_params_id,
                            self.model.parameters())
        tail_params = filter(lambda p: id(p) in  tail_params_id,
                            self.model.parameters())
        params_group = [
                        {'params': base_params, 'lr': OCFG.lr_group[0]},
                        {'params': tail_params, 'lr': OCFG.lr_group[1]},
                        {'params': self.model.model._fc.parameters(), 'lr': OCFG.lr_group[2]},
                        {'params': self.model.conv1.parameters(), 'lr': OCFG.lr_group[2]}
        ]        
        
        return params_group 

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        pass

class CustomModelClassifier(BaiscClassifier):
    def __init__(self):
        super.__init__()
        self.model = CustomModel()
        pass
        # set custom model architecture

    def _get_params_group(self):
        pass
        # set param group
        # return params_group


# ------------------
# DALI
# ------------------


class TrainPipeline(Pipeline):
    def __init__(self, image_paths, int_labels, phase='train', device_id=0):        
        super(TrainPipeline, self).__init__(DCFG.batch_size, DCFG.num_workers, device_id, exec_async=False, exec_pipelined=False, seed=42)        
        random_shuffle = True if phase == 'train' else False
        self.input = ops.readers.File(files=list(image_paths), labels=list(int_labels), random_shuffle=random_shuffle, name="Reader")
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        dali_device = 'gpu'
        self.resize = ops.Resize(device=dali_device)
        self.crop = ops.Crop(device=dali_device, crop=[224.0, 224.0], dtype=types.FLOAT)
        self.rotate = ops.Rotate(device=dali_device)  
        self.transpose = ops.Transpose(device=dali_device, perm=[2, 0, 1])
        self.phase=phase

    def define_graph(self):
        angle = fn.random.uniform(values=[0]*8 + [90.0, -90.0]) # 20% change rotate
        self.jpegs, self.labels = self.input() # (name='r')
        output = self.decode(self.jpegs)
        output = fn.python_function(output, function=dali_custom_func)
        if 'gary' in MCFG.model_type:
            output = fn.color_space_conversion(output, image_type=types.RGB, output_type=types.GRAY)
        if self.phase == 'train':
            w = fn.random.uniform(range=(224.0, 320.0))
            h = fn.random.uniform(range=(224.0, 320.0))        
            output = self.resize(output, resize_x=w, resize_y=h)        
        else:
            output = self.resize(output, size=(248.0, 248.0))
        output = self.crop(output)        
        output = self.rotate(output, angle=angle)
        output = self.transpose(output)
        output = output/255.0
        return (output, self.labels)

class LightningWrapper(DALIClassificationIterator):
    def __init__(self, *kargs, **kvargs):
        super().__init__(*kargs, **kvargs)
        self.__code__ = None
    def __next__(self):
        out = super().__next__()
        # DDP is used so only one pipeline per process
        # also we need to transform dict returned by DALIClassificationIterator to iterable
        # and squeeze the lables
        out = out[0]
        return (out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map)

class DaliEffClassifier(EfficientClassifier):    
    def __init__(self):
        super().__init__()
        self.validation_epoch_end = self.validation_epoch_end_wrapper(self.validation_epoch_end)

    def validation_epoch_end_wrapper(self, func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            self.trainer.datamodule.val_dataloader().reset()
        return wrapper
    
    def process_batch(self, batch):
        x, y = batch[0]['data'], batch[0]['label'].squeeze(-1)
        return x.float(), y.long()  

class NoisyStudentDaliEffClassifier(DaliEffClassifier):
    def __init__(self, teacher_model):
        super().__init__()
        self.teacher_model = teacher_model


    def _handle_teacher_label_logits(self, label_logits):    
        return F.softmax(label_logits / NS.teacher_softmax_temp)

    def cross_entropy_loss(self, logits, target_prob):
        target_prob = _handle_teacher_label_logits(target_prob)
        return torch.sum(target_prob*-F.log_softmax(logits))

def _get_raw_model(
    model_type=MCFG.model_type, 
    is_pretrained=MCFG.is_pretrained,
    **kwargs):    

    if 'noisy_student' in model_type:
        eff_ver = re.search("[0-9]{1}", model_type).group(0)
        dropout_rate = kwargs["dropout_rate"] if "dropout_rate" in kwargs.key() else NS.dropout_rate
        drop_connect_rate = kwargs["drop_connect_rate"] if "drop_connect_rate" in kwargs.key() else NS.drop_connect_rate
        model = EfficientNet.from_pretrained(f"efficientnet-b{eff_ver}", dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate)
    elif 'eff' in model_type:
        eff_type = re.search("b[0-7]{1}", model_type).group(0)
        raw_model = EfficientNet.from_pretrained(f"efficientnet-{eff_type}")        
    else: # model in torchvision.models 
        raw_model = getattr(torchvision.models, model_type)(pretrained=is_pretrained)
    
    print(f"Get {model_type}!")
    return raw_model

def get_model(
    model_class_name=MCFG.model_class_name,
    ckpt_path=MCFG.ckpt_path, 
    is_continued_training=MCFG.is_continued_training,    
    **kwargs):
    """
    Arguments:
        classifier_name: str, full classifer class name        
    """     
    g = globals().copy()
    model_class_names = [k for k in g.keys() if not k.startswith('_') and 'Classifier' in k]
    assert model_class_name in model_class_names, f"Wrong classifier class name, should be one of {model_class_names}"
    ModelClass = g[model_class_name]        

    model = ModelClass.load_from_checkpoint(ckpt_path) if is_continued_training else ModelClass()
    return model    

def get_pred_model(model_class_name):
    # TODO
    # best_model_ckpt = ModelFileHandler.get_best_model_ckpt()
    best_model_ckpt = MCFG.ckpt_path
    model = get_model(model_class_name=model_class_name, ckpt_path=best_model_ckpt, is_continued_training=True)
    return model
