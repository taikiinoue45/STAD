import hydra
from trainer.builder import Builder


@hydra.main(config_path='/app/github/STAD/stad/yamls/base.yaml')
def my_app(cfg):
    builder = Builder(cfg)
    builder.run_train()
    builder.run_inference()


if __name__ == "__main__":
    my_app()
