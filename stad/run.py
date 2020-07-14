import os
import hydra
import logging
from stad.trainer import Trainer


@hydra.main(config_path='/home/inoue/github/STAD/stad/yamls/mvtec.yaml')
def my_app(cfg):
    
    os.makedirs('test/normal')
    os.makedirs('test/anomaly')
    
    trainer = Trainer(cfg)
    
    if cfg.pretrained_models.school:
        trainer.load_school_pth()
    else:
        trainer.run_train_student()
        
    trainer.run_test()
    
    os.rename('.hydra', 'hydra')


if __name__ == "__main__":
    my_app()
