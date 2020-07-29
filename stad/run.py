import logging
import os
import sys
from time import time

import hydra

import stad.typehint as T
import stad.utils as U
from stad.trainer import Trainer

log = logging.getLogger(__name__)

config_path = sys.argv[1]
sys.argv.pop(1)


@hydra.main(config_path)
def my_app(cfg: T.DictConfig) -> None:

    os.rename(".hydra", "hydra")

    trainer = Trainer(cfg)

    if cfg.pretrained_models.school:
        trainer.load_school_pth()
    else:
        log.info(f"training start - {time()}")
        trainer.run_train_student()
        log.info(f"training end - {time()}")

        # Functions in stad.utils
        U.show_val_results()
        U.show_probabilistic_crop()
        U.save_loss_csv()
        U.save_training_time()

    trainer.run_test()

    # Functions in stad.utils
    U.show_test_results("normal")
    U.show_test_results("anomaly")
    U.compute_mIoU()
    U.clean_up()


if __name__ == "__main__":
    my_app()
