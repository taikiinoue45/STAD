import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import stad.datasets
import stad.models
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from stad import albu
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:

        self.cfg = cfg

        self.train_augs = self.get_augs(train_or_test="train")
        self.test_augs = self.get_augs(train_or_test="test")

        self.dataloader = {}

        self.dataloader["train"] = self.get_dataloader(
            data_dir=self.cfg.dataset.train, augs=self.train_augs
        )

        self.dataloader["val"] = self.get_dataloader(
            data_dir=self.cfg.dataset.val, augs=self.test_augs
        )

        self.dataloader["normal"] = self.get_dataloader(
            data_dir=self.cfg.dataset.test.normal, augs=self.test_augs
        )

        self.dataloader["anomaly"] = self.get_dataloader(
            data_dir=self.cfg.dataset.test.anomaly, augs=self.test_augs
        )

        self.school = self.get_school()
        self.school = self.school.to(self.cfg.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

    def get_school(self) -> nn.Module:

        return stad.models.School()

    def get_augs(self, train_or_test: str) -> albu.Compose:

        augs = []
        for i in range(len(self.cfg["augs"][train_or_test])):
            name = self.cfg["augs"][train_or_test][i]["name"]
            fn = getattr(albu, name)
            augs.append(fn(**self.cfg["augs"][train_or_test][i]["args"]))
        augs.append(ToTensorV2())

        return albu.Compose(augs)

    def get_dataloader(self, data_dir: str, augs: albu.Compose) -> DataLoader:

        Dataset = getattr(stad.datasets, self.cfg.dataset.name)

        dataset = Dataset(data_dir=Path(data_dir), augs=augs)

        dataloader = DataLoader(
            dataset=dataset, batch_size=self.cfg.train.batch_size, shuffle=True
        )
        return dataloader

    def get_optimizer(self) -> Optimizer:

        parameters = self.school.student.parameters()
        lr = self.cfg.optim.lr
        weight_decay = self.cfg.optim.weight_decay
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        return optimizer

    def get_criterion(self) -> _Loss:

        return torch.nn.MSELoss(reduction="mean")

    def load_school_pth(self) -> None:

        self.school.load_state_dict(torch.load(self.cfg.pretrained_models.school))

    def run_train_student(self) -> None:

        self.school.student.train()
        self.school.teacher.eval()

        for epoch in range(1, self.cfg.train.epochs):

            log.info(f"epoch_{epoch}")
            loss_sum = 0

            for img, _, _ in self.dataloader["train"]:
                img = img.to(self.cfg.device)
                surrogate_label, pred = self.school(img)
                loss = self.criterion(pred, surrogate_label)
                loss_sum += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss = loss_sum / len(self.dataloader["train"])
            log.info(f"loss_{epoch_loss}")

            if epoch % self.cfg.train.mini_epochs == 0:
                self.run_val(epoch)
                self.school.student.train()
                self.school.teacher.eval()

        # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
        # https://hydra.cc/docs/tutorial/working_directory
        torch.save(self.school.state_dict(), "school.pth")

    def run_val(self, epoch: int) -> None:

        self.school.teacher.eval()
        self.school.student.eval()

        cumulative_heatmap = np.array([])

        for i, (img, raw_img, mask) in enumerate(self.dataloader["val"]):

            heatmap = self.compute_heatmap(img)

            if len(cumulative_heatmap) == 0:
                cumulative_heatmap = heatmap
            else:
                cumulative_heatmap += heatmap

            # Save raw_img, mask and anomaly map
            # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
            # https://hydra.cc/docs/tutorial/working_directory
            raw_img = raw_img.squeeze().detach().numpy()
            mask = mask.squeeze().detach().numpy()

            cv2.imwrite(f"{epoch} - {i} - val_img.jpg", raw_img)
            cv2.imwrite(f"{epoch} - {i} - val_mask.png", mask)

            with open(f"{epoch} - {i} - val_heatmap.npy", "wb") as f:
                np.save(f, heatmap)

            if i + 1 == self.cfg.val.data_num:
                break

        # Update heatmap in ProbabilisticCrop
        for i, aug in enumerate(self.train_augs):

            if aug.__module__ == "stad.albu.probabilistic_crop":
                self.dataloader["train"].dataset.augs[i].heatmap = cumulative_heatmap

    def run_test(self) -> None:

        self.school.teacher.eval()
        self.school.student.eval()

        for anomaly_or_normal in ["anomaly", "normal"]:
            for i, (img, raw_img, mask) in enumerate(
                self.dataloader[anomaly_or_normal]
            ):

                heatmap = self.compute_heatmap(img)

                # Save raw_img, mask and anomaly map
                # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
                # https://hydra.cc/docs/tutorial/working_directory

                raw_img = raw_img.squeeze().detach().numpy()
                mask = mask.squeeze().detach().numpy()

                cv2.imwrite(f"{i} - test_{anomaly_or_normal}_img.jpg", raw_img)
                cv2.imwrite(f"{i} - test_{anomaly_or_normal}_mask.png", mask)

                with open(f"{i} - test_{anomaly_or_normal}_heatmap.npy", "wb") as f:
                    np.save(f, heatmap)

    def patchize(self, img: Tensor) -> Tensor:

        """
        img.shape
        B  : batch size
        C  : channels of image (same to patches.shape[1])
        iH : height of image
        iW : width of image

        pH : height of patch
        pW : width of patch
        V  : values in a patch (pH * pW * C)
        """

        B, C, iH, iW = img.shape
        pH = self.cfg.patch_size
        pW = self.cfg.patch_size

        unfold = nn.Unfold(kernel_size=(pH, pW), stride=self.cfg.test.unfold_stride)

        patches = unfold(img)  # (B, V, P)
        patches = patches.permute(0, 2, 1).contiguous()  # (B, P, V)
        patches = patches.view(-1, C, pH, pW)  # (P, C, pH, pW)
        return patches

    def compute_squared_l2_distance(
        self, pred: Tensor, surrogate_label: Tensor
    ) -> Tensor:

        losses = (pred - surrogate_label) ** 2
        losses = losses.view(losses.shape[0], -1)
        losses = torch.mean(losses, dim=1)
        losses = losses.cpu().detach()

        return losses

    def compute_heatmap(self, img: Tensor) -> np.array:

        """
        img.shape
        B  : batch size
        C  : channels of image (same to patches.shape[1])
        iH : height of image
        iW : width of image

        patches.shape
        P  : patch size
        C  : channels of image (same to img.shape[1])
        pH : height of patch
        pW : width of patch
        """

        patches = self.patchize(img)

        B, C, iH, iW = img.shape
        P, C, pH, pW = patches.shape

        heatmap = torch.zeros(P)
        quotient, remainder = divmod(P, self.cfg.test.batch_size)

        for i in tqdm(range(quotient)):

            start = i * self.cfg.test.batch_size
            end = start + self.cfg.test.batch_size

            patch = patches[start:end, :, :, :]
            patch = patch.to(self.cfg.device)

            surrogate_label, pred = self.school(patch)
            losses = self.compute_squared_l2_distance(pred, surrogate_label)
            heatmap[start:end] = losses

        patch = patches[-remainder:, :, :, :]
        patch = patch.to(self.cfg.device)

        surrogate_label, pred = self.school(patch)
        losses = self.compute_squared_l2_distance(pred, surrogate_label)
        heatmap[-remainder:] = losses

        fold = nn.Fold(
            output_size=(iH, iW),
            kernel_size=(pH, pW),
            stride=self.cfg.test.unfold_stride,
        )

        heatmap = heatmap.expand(B, pH * pW, P)
        heatmap = fold(heatmap)
        heatmap = heatmap.numpy()
        heatmap = heatmap[0, 0, :, :]

        del patches
        return heatmap
