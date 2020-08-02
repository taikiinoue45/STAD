import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import stad.datasets
import stad.models
import stad.typehint as T
from stad import albu

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: T.DictConfig) -> None:

        self.cfg = cfg

        self.train_augs = self.get_augs(train_or_test="train")
        self.test_augs = self.get_augs(train_or_test="test")

        self.dataloader = {}

        self.dataloader["pruning"] = self.get_dataloader(
            base=self.cfg.dataset.train, batch_size=1, augs=self.train_augs,
        )

        self.dataloader["train"] = self.get_dataloader(
            base=self.cfg.dataset.train,
            batch_size=self.cfg.train.batch_size,
            augs=self.train_augs,
        )

        self.dataloader["val"] = self.get_dataloader(
            base=self.cfg.dataset.val, batch_size=1, augs=self.test_augs
        )

        self.dataloader["normal"] = self.get_dataloader(
            base=self.cfg.dataset.test.normal, batch_size=1, augs=self.test_augs
        )

        self.dataloader["anomaly"] = self.get_dataloader(
            base=self.cfg.dataset.test.anomaly, batch_size=1, augs=self.test_augs
        )

        self.is_pruning = None
        self.school = self.get_school()
        self.school = self.school.to(self.cfg.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

    def get_school(self) -> T.Module:

        return stad.models.School()

    def get_augs(self, train_or_test: str) -> T.Compose:

        return albu.load(self.cfg[train_or_test]["augs"], data_format="yaml")

    def get_dataloader(self, base: str, batch_size: int, augs: T.Compose) -> T.DataLoader:

        Dataset = getattr(stad.datasets, self.cfg.dataset.name)
        dataset = Dataset(base=Path(base), augs=augs)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def get_optimizer(self) -> T.Optimizer:

        parameters = self.school.student.parameters()
        lr = self.cfg.train.optim.lr
        weight_decay = self.cfg.train.optim.weight_decay
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        return optimizer

    def get_criterion(self) -> T.Loss:

        return torch.nn.MSELoss(reduction="mean")

    def load_school_pth(self) -> None:

        self.school.load_state_dict(torch.load(self.cfg.train.pretrained.school))

    def run_train_student(self) -> None:

        self.school.student.train()
        self.school.teacher.eval()

        pbar = tqdm(range(1, self.cfg.train.epochs), desc="train")
        for epoch in pbar:

            log.info(f"epoch - {epoch}")
            loss_sum = 0

            for sample in self.dataloader["train"]:
                img = sample["image"].to(self.cfg.device)
                surrogate_label, pred = self.school(img)
                loss = self.criterion(pred, surrogate_label)
                loss_sum += loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss = loss_sum / len(self.dataloader["train"])
            log.info(f"loss - {epoch_loss}")

            if epoch % (self.cfg.train.epochs // 10) == 0:
                self.run_val(epoch)
                self.school.student.train()
                self.school.teacher.eval()

        if self.is_pruning:
            self.run_pruning()
        else:
            torch.save(self.school.state_dict(), "school.pth")

    def run_pruning(self) -> None:

        li = []
        pbar = tqdm(self.dataloader["pruning"], desc="pruning")
        for sample in pbar:
            img = sample["image"].to(self.cfg.device)
            img_path = sample["img_path"][0]
            heatmap = self.compute_heatmap(img)
            li.append([heatmap.max(), img_path])

        threshold = int(self.cfg.train.pruning_rate * len(self.dataloader["train"].dataset))
        li = sorted(li, key=lambda x: [0], reverse=True)
        li = li[:threshold]
        for _, img_path in li:
            self.dataloader["train"].dataset.img_paths.remove(img_path)
            log.info(f"pruned data - {img_path}")

        # Initialize student network and optimizer
        self.school.initialize_student()
        self.school.student.to(self.cfg.device)
        self.optimizer = self.get_optimizer()

    def run_val(self, epoch: int) -> None:

        self.school.teacher.eval()
        self.school.student.eval()
        cumulative_heatmap = np.array([])

        pbar = tqdm(self.dataloader["val"], desc="val")
        for i, sample in enumerate(pbar):

            img = sample["image"]
            stem = Path(sample["img_path"][0]).stem
            heatmap = self.compute_heatmap(img)

            if len(cumulative_heatmap) == 0:
                cumulative_heatmap = heatmap
            else:
                cumulative_heatmap += heatmap

            if not self.is_pruning:
                with open(f"{epoch} - {i} - val - {stem}.npy", "wb") as f:
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
            pbar = tqdm(self.dataloader[anomaly_or_normal], desc=f"test - {anomaly_or_normal}")
            for i, sample in enumerate(pbar):

                img = sample["image"]
                stem = Path(sample["img_path"][0]).stem
                heatmap = self.compute_heatmap(img)

                # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
                # https://hydra.cc/docs/tutorial/working_directory
                with open(f"{i} - test_{anomaly_or_normal} - {stem}.npy", "wb") as f:
                    np.save(f, heatmap)

    def patchize(self, img: T.Tensor) -> T.Tensor:

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

    def compute_squared_l2_distance(self, pred: T.Tensor, surrogate_label: T.Tensor) -> T.Tensor:

        losses = (pred - surrogate_label) ** 2
        losses = losses.view(losses.shape[0], -1)
        losses = torch.mean(losses, dim=1)
        losses = losses.cpu().detach()

        return losses

    def compute_heatmap(self, img: T.Tensor) -> T.NDArray[(T.Any, T.Any), float]:

        """
        img.shape -> (B, C, iH, iW)
        B  : batch size
        C  : channels of image (same to patches.shape[1])
        iH : height of image
        iW : width of image

        patches.shape -> (P, C, pH, pW)
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

        for i in range(quotient):

            start = i * self.cfg.test.batch_size
            end = start + self.cfg.test.batch_size

            patch = patches[start:end, :, :, :]  # (self.cfg.test.batch_size, C, pH, pW)
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
            output_size=(iH, iW), kernel_size=(pH, pW), stride=self.cfg.test.unfold_stride,
        )

        heatmap = heatmap.expand(B, pH * pW, P)
        heatmap = fold(heatmap)
        heatmap = heatmap.squeeze().numpy()

        del patches
        return heatmap
