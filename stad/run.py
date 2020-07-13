import os
import hydra
import logging
from stad.trainer import Trainer


@hydra.main(config_path='/dgx/github/STAD/stad/yamls/somic.yaml')
def my_app(cfg):
    
    os.mkdir('normal')
    os.mkdir('anomaly')
    
    trainer = Trainer(cfg)
    
    if cfg.pretrained_models.school:
        trainer.load_school_pth()
    else:
        trainer.run_train_student()
        
    trainer.run_test()


if __name__ == "__main__":
    my_app()
