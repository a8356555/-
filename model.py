# model.py
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
from torchvision import models
from .config import mcfg, dcfg, ocfg


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
            !nvidia-smi        
       
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
        if ocfg.has_differ_lr:
            params_group = self._get_params_group()
            print(params_group)

            optimizer = torch.optim.Adam(params_group, weight_decay=ocfg.weight_decay) if ocfg.optim_name == 'Adam' else \
                        torch.optim.SGD(params_group, momentum=ocfg.momentum, weight_decay=ocfg.weight_decay)

        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=ocfg.lr, weight_decay=ocfg.weight_decay) if ocfg.optim_name == 'Adam' else \
                        torch.optim.SGD(self.model.parameters(), lr=ocfg.lr, momentum=ocfg.momentum, weight_decay=ocfg.weight_decay)

        if ocfg.has_scheduler:
            if ocfg.schdlr_name == 'OneCycleLR':
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=ocfg.max_lr, total_steps=ocfg.total_steps)  # 设置学习率下降策略

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
        self.model = _get_raw_model()
        num_input_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_input_fts, dcfg.class_num)        
    
    def _get_params_group(self):
        fc_params = list(map(id, self.model.fc.parameters()))
        layer4_params = list(map(id, self.model.layer4.parameters()))
        all_params = list(map(id, self.model.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params + layer4_params,
                            self.model.parameters())
        params_group = [
                        {'params': base_params, 'lr': ocfg.lr_group[0]},
                        {'params': self.model.layer4.parameters(), 'lr': ocfg.lr_group[1]},
                        {'params': self.model.fc.parameters(), 'lr': ocfg.lr_group[2]}
        ]                
        return params_group
        
class EfficientClassifier(YuShanClassifier):
    def __init__(self):
        super().__init__()
        self.model = _get_raw_model()
        num_input_fts = self.model._fc.in_features
        self.model._fc = nn.Linear(num_input_fts, dcfg.class_num)        
    
    def _get_params_group(self):
        fc_params_id = list(map(id, self.model._fc.parameters()))
        tail_params_id = list(map(id, self.model._blocks[-3:].parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id + tail_params_id,
                            self.model.parameters())
        tail_params = filter(lambda p: id(p) in  tail_params_id,
                            self.model.parameters())
        params_group = [
                        {'params': base_params, 'lr': ocfg.lr_group[0]},
                        {'params': tail_params, 'lr': ocfg.lr_group[1]},
                        {'params': self.model._fc.parameters(), 'lr': ocfg.lr_group[2]}
        ]                
        return params_group

class GrayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, padding=1)
        self.model = _get_raw_model()
    def forward(self, x):
        x = self.conv1(x)        
        return self.model(x)

class GrayResClassifier(YuShanClassifier):
    def __init__(self):
        super().__init__()
        self.model = GrayModel()
        num_input_fts = self.model.model.fc.in_features
        self.model.model.fc = nn.Linear(num_input_fts, dcfg.class_num)        
                    
    def _get_params_group(self):
        conv1_params_id = list(map(id, self.model.conv1.parameters())) 
        fc_params_id = list(map(id, self.model.model.fc.parameters()))
        layer4_params_id = list(map(id, self.model.model.layer4.parameters()))
        # all_params = list(map(id, self.model.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id + layer4_params_id + conv1_params_id,
                            self.model.parameters())
        params_group = [
                        {'params': base_params, 'lr': ocfg.lr_group[0]},
                        {'params': self.model.model.layer4.parameters(), 'lr': ocfg.lr_group[1]},
                        {'params': self.model.model.fc.parameters(), 'lr': ocfg.lr_group[2]},
                        {'params': self.model.conv1.parameters(), 'lr': ocfg.lr_group[2]}
        ]        
        return params_group 

class GrayEffClassifier(YuShanClassifier):
    def __init__(self):
        super().__init__()
        self.model = GrayModel()
        num_input_fts = self.model.model._fc.in_features
        self.model.model._fc = nn.Linear(num_input_fts, dcfg.class_num)        
                    
    def _get_params_group(self):
        con1_params_id = list(map(id, self.model.conv1.parameters())) 
        fc_params_id = list(map(id, self.model.model._fc.parameters()))
        tail_params_id = list(map(id, self.model.model._blocks[-3:].parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id + tail_params_id + con1_params_id,
                            self.model.parameters())
        tail_params = filter(lambda p: id(p) in  tail_params_id,
                            self.model.parameters())
        params_group = [
                        {'params': base_params, 'lr': ocfg.lr_group[0]},
                        {'params': tail_params, 'lr': ocfg.lr_group[1]},
                        {'params': self.model.model._fc.parameters(), 'lr': ocfg.lr_group[2]},
                        {'params': self.model.conv1.parameters(), 'lr': ocfg.lr_group[2]}
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
        self.model = CustomModel()
        pass
        # set custom model architecture

    def _get_params_group(self):
        pass
        # set param group
        # return params_group

def _get_raw_model():    
    if '18' in mcfg.model_type:
        print('get res18!')
        raw_model = models.resnet18(pretrained=mcfg.is_pretrained)    
    elif '34' in mcfg.model_type:
        print('get res34!')
        raw_model = models.resnet34(pretrained=mcfg.is_pretrained)
    elif '50' in mcfg.model_type:
        print('get res50!')
        raw_model = models.resnet50(pretrained=mcfg.is_pretrained)
    elif 'b0' in mcfg.model_type:
        print('get eff-net-b0!')
        raw_model = EfficientNet.from_pretrained('efficientnet-b0')
    elif 'b1' in mcfg.model_type:
        print('get eff-net-b1!')
        raw_model = EfficientNet.from_pretrained('efficientnet-b1')
    elif 'b2' in mcfg.model_type:
        print('get eff-net-b2!')
        raw_model = EfficientNet.from_pretrained('efficientnet-b2')
    # elif ...
    return raw_model

def get_model():    
    if 'res' in mcfg.model_type:
        model = ResClassifier().load_from_checkpoint(mcfg.ckpt_path) if mcfg.is_continued else ResClassifier()
    elif 'eff' in mcfg.model_type:
        model = EfficientClassifier().load_from_checkpoint(mcfg.ckpt_path) if mcfg.is_continued else EfficientClassifier()
    elif 'res' in mcfg.model_type and 'gray' in mcfg.model_type:
        model = GrayResClassifier().load_from_checkpoint(mcfg.ckpt_path) if mcfg.is_continued else GrayResClassifier()
    elif 'eff' in mcfg.model_type and 'gray' in mcfg.model_type:
        model = GrayEffClassifier().load_from_checkpoint(mcfg.ckpt_path) if mcfg.is_continued else GrayEffClassifier()
    elif 'custom' in mcfg.model_type:
        model = CustomModelClassifier.load_from_checkpoint(mcfg.ckpt_path) if mcfg.is_continued else CustomModelClassifier()
    # elif ...
    else:
        raise RuntimeError("invalid model type config")
    return model