import os
import sys

import hydra
import mlflow
from omegaconf import DictConfig

from stad.runner import Runner


mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/inoue@nablas.com/stad")

config_path = sys.argv[1]
sys.argv.pop(1)


@hydra.main(config_path)
def main(cfg: DictConfig) -> None:

    os.rename(".hydra", "hydra")

    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
