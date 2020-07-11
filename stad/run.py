import hydra
from stad.trainer import Trainer


@hydra.main(config_path='/dgx/github/STAD/stad/yamls/base.yaml')
def my_app(cfg):
    trainer = Trainer(cfg)
    trainer.run_train()
    trainer.run_test()


if __name__ == "__main__":
    my_app()
