import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .model import get_model
from .config import DCFG, MCFG, OCFG, NS
CFGs = [MCFG, DCFG, OCFG, NS]
from .utils import ConfigHandler

def single_train(model, datamodule, is_for_testing=False, is_user_input_needed=True):
    if is_for_testing:
        trainer = pl.Trainer(
            max_epochs=MCFG.max_epochs, 
            gpus=MCFG.gpus, 
            precision=MCFG.precision if MCFG.is_apex_used else None,
            amp_level=MCFG.amp_level if MCFG.is_apex_used else None,        
        )
    else:
        ConfigHandler.save_config(CFGs, folder=MCFG.target_version_folder, model=model, is_user_input_needed=is_user_input_needed)
        logger = TensorBoardLogger(MCFG.root_model_folder, name=MCFG.model_type, version=MCFG.version)

        checkpoint_callback = ModelCheckpoint(
            monitor=MCFG.monitor,
            dirpath=MCFG.target_version_folder / "checkpoints",
            filename='{epoch}',
            save_top_k = MCFG.save_top_k_models,
            every_n_val_epochs=MCFG.save_every_n_epoch,
        )

        trainer = pl.Trainer(
            logger=logger,        
            max_epochs=MCFG.max_epochs, 
            gpus=MCFG.gpus, 
            precision=MCFG.precision if MCFG.is_apex_used else None,
            amp_level=MCFG.amp_level if MCFG.is_apex_used else None,
            log_every_n_steps=MCFG.log_every_n_steps, 
            flush_logs_every_n_steps=MCFG.log_every_n_steps,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=MCFG.ckpt_path if MCFG.is_continued_training else None
        )

    trainer.fit(model, datamodule=datamodule)
    return trainer, model

def multi_train(config_dicts, model_classes, datamodules):
    """
    Arguments:
        configs: list, a list of config dict, eg. [{'batch_size': 128}, {'batch_size': 256}]
        model_classes: list, a list of model class, eg. [DaliClassifier, YushanClassifier]
    """
    assert len(config_dicts) == len(datamodules) == len(model_classes), "unmatched input numbers"    
    for config_dict, Model, datamodule in zip(config_dicts, model_classes, datamodules):
        ConfigHandler.change_CFGs(CFGs, **config_dict)
        model = Model()
        single_train(model, datamodule, is_user_input_needed=False)

        
def train():
    model = get_model()        
    datamodule = create_datamodule()
    trainer, model = single_train(model, datamodule)    
    return model, trainer, datamodule