import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .model import get_model
from .config import dcfg, mcfg, ocfg, save_config, change_config

def single_train(model, datamodule, is_for_testing=False, is_user_input_needed=True):
    if is_for_testing:
        trainer = pl.Trainer(
            max_epochs=mcfg.max_epochs, 
            gpus=mcfg.gpus, 
            precision=mcfg.precision if mcfg.is_apex_used else None,
            amp_level=mcfg.amp_level if mcfg.is_apex_used else None,        
        )
    else:
        save_config(folder_path=mcfg.model_folder_path, model=model, is_user_input_needed=is_user_input_needed)
        logger = TensorBoardLogger(mcfg.model_folder_path)


        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=mcfg.model_folder_path / "checkpoints",
            filename='{epoch}',
            save_top_k = mcfg.save_top_k_models,
            every_n_val_epochs=mcfg.save_every_n_epoch,
        )

        trainer = pl.Trainer(
            logger=logger,        
            max_epochs=mcfg.max_epochs, 
            gpus=mcfg.gpus, 
            precision=mcfg.precision if mcfg.is_apex_used else None,
            amp_level=mcfg.amp_level if mcfg.is_apex_used else None,
            log_every_n_steps=mcfg.log_every_n_steps, 
            flush_logs_every_n_steps=mcfg.log_every_n_steps,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=mcfg.ckpt_path if mcfg.is_continued else None
        )

    trainer.fit(model, datamodule=datamodule)
    return trainer, model

def multi_train(config_dicts, model_classes, datamodules):
    """
    Arguments:
        configs: a list of config dict, eg. [{'batch_size': 128}, {'batch_size': 256}]
        model_classes: a list of model class, eg. [DaliClassifier, YushanClassifier]
    """
    assert len(config_dicts) == len(datamodules) == len(model_classes), "unmatched input numbers"    
    for config_dict, Model, datamodule in zip(config_dicts, model_classes, datamodules):
        change_config(**config_dict)
        model = Model()
        single_train(model, datamodule, is_user_input_needed=False)

        
def train(args):
    model = get_model()        
    datamodule = create_datamodule(args)    
    logger = TensorBoardLogger(mcfg.logger_path, name=mcfg.model_type, version=mcfg.version)

    save_config(folder_path=mcfg.model_folder_path, model=model)
    
    checkpoint_callback = ModelCheckpoint(
        monitor=mcfg.monitor,
        dirpath=os.path.join(mcfg.logger_path, mcfg.model_type, mcfg.version, mcfg.model_ckpt_dirname),
        filename='{epoch}',
        save_top_k = mcfg.save_top_k_models,
        every_n_val_epochs=mcfg.save_every_n_epoch,
    )

    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=mcfg.max_epochs, 
        gpus=mcfg.gpus, 
        precision=mcfg.precision if mcfg.is_apex_used else None,
        amp_level=mcfg.amp_level if mcfg.is_apex_used else None,
        log_every_n_steps=mcfg.log_every_n_steps, 
        flush_logs_every_n_steps=mcfg.log_every_n_steps,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=mcfg.ckpt_path if mcfg.is_continued else None
    )

    trainer.fit(model, datamodule)
    return model, trainer, datamodule