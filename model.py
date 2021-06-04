# model.py
import pytorch_lightning as pl
from config import mcfg, dcfg, ocfg
import torch
from torch import nn

class YuShanClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()        
        self.model = mcfg.raw_model
        num_input_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_input_fts, mcfg.pred_size)
        self.time = 0
        
    def forward(self, x):
        return self.model(x)

    def process_batch(self, batch):
      x, y = batch
      return x.float(), y.long()

    def cross_entropy_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)

    def training_step(self, train_batch, batch_idx):
        if batch_idx == 1:
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
        loss = sum([x['loss'] for x in outputs])/len(self.trainer.datamodule.train_dataloader())

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
        loss = sum([x['loss'] for x in outputs])/len(self.trainer.datamodule.val_dataloader())

        print(f'- val_epoch_acc: {acc}, val_loss: {loss}\n')        
        self.log('val_epoch_acc', acc)    

    def configure_optimizers(self):
        if ocfg.has_differ_lr:
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
            
            optimizer = torch.optim.Adam(params_group, weight_decay=ocfg.weight_decay) if ocfg.optim_name == 'Adam' else \
                        torch.optim.SGD(params_group, momentum=ocfg.momentum, weight_decay=ocfg.weight_decay)

        else:
            optimizer = torch.optim.Adam(lr=ocfg.lr, weight_decay=ocfg.weight_decay) if ocfg.optim_name == 'Adam' else \
                        torch.optim.SGD(lr=ocfg.lr, momentum=ocfg.momentum, weight_decay=ocfg.weight_decay)

        if ocfg.has_scheduler:
            if ocfg.schdlr_name == 'OneCycleLR':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=ocfg.max_lr, total_steps=ocfg.total_steps)  # 设置学习率下降策略

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            }
        else:
            return optimizer

class DaliYuShanClassifier(YuShanClassifier):
    def __init__(self, train_input, valid_input):
        super().__init__()
        self.train_input = train_input
        self.valid_input = valid_input 
        
    # def prepare_data():
    #     self.pip_train = CustomPipeline(self.train_input['path'], self.train_input['int_label'], dcfg.batch_size)
    #     self.pip_train.build()
    #     print('train built')
    #     self.train_loader = DALIClassificationIterator(self.pip_train, size=pip_train.epoch_size("r"))

    #     self.pip_valid = CustomPipeline(self.valid_input['path'], self.valid_input['int_label'], dcfg.batch_size)
    #     self.pip_valid.build()
    #     print('valid built')
    #     self.valid_loader = DALIClassificationIterator(self.pip_valid, size=pip_valid.epoch_size("r"))        
        
    def train_dataloader(self):
        # return self.train_loader
        return dali_iter_train
    
    def val_dataloader(self):
        # return self.valid_loader
        return dali_iter_train

    def process_batch(self, batch):
        x = batch[0]["data"]
        y = batch[0]["label"].squeeze(-1).long().cuda()
        return (x, y)