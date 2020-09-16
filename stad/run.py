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
def main(cfg: T.DictConfig) -> None:

    os.rename(".hydra", "hydra")

    trainer = Trainer(cfg)

    log.info(f"training start - {time()}")
    trainer.run_train_student()
    log.info(f"training end - {time()}")
    U.show_val_results(cfg)
    U.show_probabilistic_crop(cfg)
    U.save_loss_csv()
    U.save_training_time()

    trainer.run_test()
    U.show_test_results(cfg)
    U.compute_mIoU(cfg)
    U.clean_up()


if __name__ == "__main__":
    main()
