import os

import hydra

import stad.typehint as T
import stad.utils as U
from stad.trainer import Trainer


@hydra.main(config_path="/dgx/github/STAD_dev/stad/yamls/mvtec.yaml")
def my_app(cfg: T.DictConfig) -> None:

    print(cfg.pretty())
    os.rename(".hydra", "hydra")

    trainer = Trainer(cfg)

    if cfg.pretrained_models.school:
        trainer.load_school_pth()
    else:
        trainer.run_train_student()

        # Functions in stad.utils
        U.show_val_results()
        U.show_probabilistic_crop()
        U.save_loss_csv()

    trainer.run_test()

    # Functions in stad.utils
    U.show_test_results("normal")
    U.show_test_results("anomaly")
    U.compute_mIoU()
    U.clean_up()


if __name__ == "__main__":
    my_app()
