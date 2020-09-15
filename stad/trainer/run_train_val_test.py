from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import stad.typehint as T


class TrainerRunTrainValTest:

    school: T.Module
    cfg: T.DictConfig
    log: T.Logger
    dataloader: T.DataLoader
    criterion: T.Loss
    optimizer: T.Optimizer
    augs: T.Compose

    def run_train_student(self) -> None:

        self.school.student.train()
        self.school.teacher.eval()

        pbar = tqdm(range(1, self.cfg.train.epochs), desc="train")
        for epoch in pbar:

            self.log.info(f"epoch - {epoch}")
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
            self.log.info(f"loss - {epoch_loss}")

            if epoch % (self.cfg.train.epochs // 10) == 0:
                self.run_val(epoch)
                self.school.student.train()
                self.school.teacher.eval()

        torch.save(self.school.state_dict(), "school.pth")

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

            with open(f"{epoch} - {i} - val - {stem}.npy", "wb") as f:
                np.save(f, heatmap)

            if i + 1 == self.cfg.val.data_num:
                break

        # Update heatmap in ProbabilisticCrop
        for i, aug in enumerate(self.augs["train"]):
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
            output_size=(iH, iW),
            kernel_size=(pH, pW),
            stride=self.cfg.test.unfold_stride,
        )

        heatmap = heatmap.expand(B, pH * pW, P)
        heatmap = fold(heatmap)
        heatmap = heatmap.squeeze().numpy()

        del patches
        return heatmap
