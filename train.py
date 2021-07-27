import os
from pytorch_lightning.callbacks import ModelCheckpoint

from .model import get_model
from .config import dcfg, mcfg, ocfg
from .utils import save_config

def train(args):
    model = get_model()        
    data_module = create_datamodule(args)    
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

    trainer.fit(model, data_module)
    return model, trainer, data_module