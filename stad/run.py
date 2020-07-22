import os

import hydra
from omegaconf import DictConfig
from stad.trainer import Trainer
from stad.utils import *


@hydra.main(config_path="/app/github/STAD/stad/yamls/mvtec.yaml")
def my_app(cfg: DictConfig) -> None:

    print(cfg.pretty())
    os.rename(".hydra", "hydra")

    trainer = Trainer(cfg)

    if cfg.pretrained_models.school:
        trainer.load_school_pth()
    else:
        trainer.run_train_student()

        # Functions in stad.utils
        show_val_results()
        show_probabilistic_crop()
        save_loss_csv()

    trainer.run_test()

    # Functions in stad.utils
    show_test_results("normal")
    show_test_results("anomaly")
    compute_mIoU()
    clean_up()


if __name__ == "__main__":
    my_app()
