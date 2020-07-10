import hydra
from stad.trainer import Trainer


@hydra.main(config_path='/home/inoue/inoue/github/STAD/stad/yamls/base_dgx.yaml')
def my_app(cfg):
    trainer = Trainer(cfg)
    trainer.run_train()
    trainer.run_inference()


if __name__ == "__main__":
    my_app()
