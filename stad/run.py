import os
import sys
import hydra
import logging

from pathlib import Path
from stad.trainer import Trainer
from stad.utils import *


@hydra.main(config_path='/app/github/STAD/stad/yamls/mvtec.yaml')
def my_app(cfg):
    
    print(cfg.pretty())
    
    # Prepare the folders to store img, mask and anomaly map
    os.makedirs('test/normal')
    os.makedirs('test/anomaly')
    os.makedirs('val')
    os.rename('.hydra', 'hydra')
    os.rename(os.getcwd(), os.getcwd()+f'_{cfg.experiment}')
    
    trainer = Trainer(cfg)
    
    if cfg.pretrained_models.school:
        trainer.load_school_pth()
    else:
        trainer.run_train_student()
        
        # Functions in stad.utils
        show_val_results(Path('./val'))
        show_probabilistic_crop()
        show_losses()

    trainer.run_test()
    
    # Functions in stad.utils
    show_test_results(Path('./test/anomaly'))
    show_test_results(Path('./test/normal'))
    compute_mIoU()
    clean_up()
    
    
    
    
if __name__ == "__main__":
    my_app()
