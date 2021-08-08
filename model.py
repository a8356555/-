# model.py
import os
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
from torchvision import models
from .config import MCFG, DCFG, OCFG, NS
from .preprocess import dali_custom_func

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch as dalitorch
import torch.utils.dlpack as torch_dlpack

class YuShanClassifier(pl.LightningModule):
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
        if batch_idx == 1:
            print(f'current epoch: {self.current_epoch}')
            #os.system("nvidia-smi")        
       
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

        print(f'- train_epoch_acc: {acc}, train_loss: {loss}\n')
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

        print(f'- val_epoch_acc: {acc}, val_loss: {loss}\n')        
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

class ResNetClassifier(YuShanClassifier):
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
        
class EfficientClassifier(YuShanClassifier):
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

class GrayResClassifier(YuShanClassifier):
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

class GrayEffClassifier(YuShanClassifier):
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

class CustomModelClassifier(YuShanClassifier):
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

    def validation_epoch_end(self, outputs):
        epoch_corrects = sum([x['running_corrects'] for x in outputs])
        dataset_size = sum([x['batch_size'] for x in outputs])
        acc = epoch_corrects/dataset_size  
        loss = sum([x['loss'] for x in outputs])/dataset_size

        print(f'- val_epoch_acc: {acc}, val_loss: {loss}\n')        
        self.log('val_epoch_acc', acc)                 
        self.trainer.datamodule.val_dataloader().reset()

    def process_batch(self, batch):
        x, y = batch[0]['data'], batch[0]['label'].squeeze(-1)
        return x.float(), y.long()  

class NoisyStudentDaliEffClassifier(DaliEffClassifier):
    def __init__(self):
        super().__init__()

    def _handle_teacher_label_logits(self, label_logits):    
        return F.softmax(label_logits / NS.teacher_softmax_temp)

    def cross_entropy_loss(self, logits, target_prob):
        target_prob = _handle_teacher_label_logits(target_prob)
        return torch.sum(target_prob*-F.log_softmax(logits))


def load_target_ckpt_model(model_type, version, )
    target_model_version_folder = MCFG.

def _get_raw_model():    
    if '18' in MCFG.model_type:
        print('get res18!')
        raw_model = models.resnet18(pretrained=MCFG.is_pretrained)    
    elif '34' in MCFG.model_type:
        print('get res34!')
        raw_model = models.resnet34(pretrained=MCFG.is_pretrained)
    elif '50' in MCFG.model_type:
        print('get res50!')
        raw_model = models.resnet50(pretrained=MCFG.is_pretrained)
    elif 'b0' in MCFG.model_type:
        print('get eff-net-b0!')
        raw_model = EfficientNet.from_pretrained('efficientnet-b0')
    elif 'b1' in MCFG.model_type:
        print('get eff-net-b1!')
        raw_model = EfficientNet.from_pretrained('efficientnet-b1')
    elif 'b2' in MCFG.model_type:
        print('get eff-net-b2!')
        raw_model = EfficientNet.from_pretrained('efficientnet-b2')
    elif 'noisy_student' in MCFG.model_type:
        model = EfficientNet.from_pretrained(f"efficientnet-b{NS.student_iter}", dropout_rate=NS.dropout_rate, drop_connect_rate=NS.drop_connect_rate)
    # elif ...
    return raw_model

def get_model(): 
    #TODO: adding dali   
    if 'res' in MCFG.model_type:
        model = ResClassifier().load_from_checkpoint(MCFG.ckpt_path) if MCFG.is_continued else ResClassifier()
    elif 'eff' in MCFG.model_type:
        model = EfficientClassifier().load_from_checkpoint(MCFG.ckpt_path) if MCFG.is_continued else EfficientClassifier()
    elif 'res' in MCFG.model_type and 'gray' in MCFG.model_type:
        model = GrayResClassifier().load_from_checkpoint(MCFG.ckpt_path) if MCFG.is_continued else GrayResClassifier()
    elif 'eff' in MCFG.model_type and 'gray' in MCFG.model_type:
        model = GrayEffClassifier().load_from_checkpoint(MCFG.ckpt_path) if MCFG.is_continued else GrayEffClassifier()
    elif 'custom' in MCFG.model_type:
        model = CustomModelClassifier.load_from_checkpoint(MCFG.ckpt_path) if MCFG.is_continued else CustomModelClassifier()
    elif 'noisy_student' in MCFG.model_type:
        model = NoisyStudentDaliEffClassifier.load_from_checkpoint(MCFG.ckpt_path) if MCFG.is_continued else NoisyStudentDaliEffClassifier()
    # elif ...
    else:
        raise RuntimeError("invalid model type config")
    return model