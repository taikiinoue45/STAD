import math
import os

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray as NDArray
from torch import Tensor
from tqdm import tqdm

from stad.metrics import compute_auroc
from stad.runner import BaseRunner
from stad.utils import savefig


class Runner(BaseRunner):
    def run(self) -> None:

        pbar = tqdm(range(self.cfg.runner.epochs), desc="epochs")
        for epoch in pbar:
            os.makedirs(f"epochs/{epoch}")
            self._train(epoch)
            self._validate(epoch)
        self._test()

    def _train(self, epoch: int) -> None:

        self.school.student.train()
        self.school.teacher.eval()
        loss_list = []
        for _, img, mask in self.dataloader_dict["train"]:
            img = img.to(self.cfg.runner.device)
            student_pred, teacher_pred = self.school(img)
            loss = self.criterion(student_pred, teacher_pred)
            loss_list.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_loss = sum(loss_list) / len(loss_list)
        mlflow.log_metric("Train Loss", train_loss, step=epoch)

    def _validate(self, epoch: int) -> None:

        self.school.student.eval()
        self.school.teacher.eval()
        all_img_paths = []
        all_heatmaps = []
        all_masks = []
        for img_path, img, mask in self.dataloader_dict["val"]:

            assert len(img) == 1  # num_batch should be 1 in val dataset
            heatmap = self._compute_heatmap(img)
            all_img_paths.append(img_path[0])
            all_heatmaps.append(heatmap)
            all_masks.append(mask.squeeze().detach().numpy())

        auroc = compute_auroc(epoch, np.array(all_heatmaps), np.array(all_masks))
        mlflow.log_metric("AUROC", auroc, step=epoch)

        savefig(epoch, all_img_paths, all_heatmaps, all_masks)

    def _test(self) -> None:

        pass

    def _compute_heatmap(self, img: Tensor) -> NDArray:
        """
        img.shape -> (B, C, iH, iW)
        B  : Number of image batches. It should be 1.
        C  : Channels of image (same to patches.shape[1])
        iH : Height of image
        iW : Width of image

        patches.shape -> (P, C, pH, pW)
        P  : Number of patches
        C  : Channels of image (same to img.shape[1])
        pH : Height of patch
        pW : Width of patch
        """

        all_patches = self._patchize(img)

        B, C, iH, iW = img.shape
        P, C, pH, pW = all_patches.shape

        heatmap = torch.zeros(P)
        num_patch_batches = self.cfg.runner.num_patch_batches

        for i in range(math.ceil(P / num_patch_batches)):

            start = i * num_patch_batches
            end = min(P, start + num_patch_batches)

            mb_patches = all_patches[start:end, :, :, :]  # mb_patches.shape = (B, C, pH, pW)
            mb_patches = mb_patches.to(self.cfg.runner.device)

            mb_student_pred, mb_teacher_pred = self.school(mb_patches)
            loss = self.criterion(mb_student_pred, mb_teacher_pred)
            heatmap[start:end] = loss.item()

        heatmap = heatmap.expand(B, pH * pW, P)
        heatmap = F.fold(
            heatmap,
            output_size=(iH, iW),
            kernel_size=(pH, pW),
            stride=self.cfg.runner.unfold_stride,
        )
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        return heatmap

    def _patchize(self, img: Tensor) -> Tensor:
        """
        B  : Batch size
        C  : Channels of image (same to patches.shape[1])
        iH : Height of image
        iW : Width of image
        pH : Height of patch
        pW : Width of patch
        V  : Number of elements in a patch (pH * pW * C)
        P  : Number of patches
        """

        B, C, iH, iW = img.shape
        pH = self.cfg.runner.patch_size
        pW = self.cfg.runner.patch_size
        stride = self.cfg.runner.unfold_stride

        patches = F.unfold(img, kernel_size=(pH, pW), stride=stride)  # patches.shape -> (B, V, P)
        patches = patches.permute(0, 2, 1).contiguous()  # patches.shape -> (B, P, V)
        patches = patches.view(-1, C, pH, pW)  # patches.shape -> (P, C, pH, pW)
        return patches
