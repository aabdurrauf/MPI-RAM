# .venv310/Scripts/python ram/mpi/finetune_mpi_full.py

import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import deepinv as dinv

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# from utils.mpiSuperResUtils import *
from ram import finetune
from utils.mpiSuperResPhysics import MPISuperResPhysics
from utils.mpiDataset import MpiDataset
from utils.loadData import *
from ram.models.ram import RAM

visualize = False
device = "cuda"
useGPUno = 0
display_some_samples = True

# manually assigned, could be done via command params
bs = 64
step_num = 10e4 # 20e4
scale_factor = 2
file_dir = "ram/mpi/data/train/"
useAugmentation = False

snrThreshold = 5
n1 = n2 = 32
torch.cuda.set_device(useGPUno)

# get dataset
trainDS, copymax, copymin, epoch_nb = get_train_data(file_dir, scale_factor, bs, step_num, n1, n2, snrThreshold, useAugmentation)

# model init
model_ram = RAM(device=device)

print(f"\nRunning RAM fine-tuning for scale x{scale_factor}...\n")

y = trainDS.LR.to(device)
x_gt = trainDS.HR.to(device)

trainDS_bs = MpiDataset(
    x_gt.clone(),
    y.clone(),
    maxhr=copymax,
    minhr=copymin,
    batch_size=bs,
    up=0.7,
    down=0.15
)

physics = MPISuperResPhysics(
    dataset=trainDS_bs,
    scale_factor=scale_factor
)

model = finetune(model_ram, y, physics, early_stop=False, transform=None, max_iter=2, noise_loss='noiseless', lr=1e-5, batch_size=bs)

# below code throws out of memory error in my laptop :(
# run inference
with torch.no_grad():
    x_hat = model(y, physics=physics)

if display_some_samples:
    with torch.no_grad():
        hr_from_A = physics.A(x_gt)
        adjoint_fn = getattr(physics, "A_adjoint", None) or getattr(physics, "A_djoint")
        lr_from_adjoint = adjoint_fn(y)
        hr_A_then_adjoint = adjoint_fn(hr_from_A)

    def _to_display_batch(t, n=4):
        t = t.detach().cpu()
        if torch.is_complex(t):
            t = torch.abs(t)

        imgs = []
        for i in range(min(n, t.shape[0])):
            s = t[i]
            if s.ndim == 3:  # [C, H, W]
                if s.shape[0] > 1:
                    s = torch.linalg.norm(s, dim=0)
                else:
                    s = s[0]
            imgs.append(s.numpy())
        return imgs

    samples = [
        _to_display_batch(y, 4),                 # LR
        _to_display_batch(x_gt, 4),              # HR
        _to_display_batch(hr_from_A, 4),         # A(HR)
        _to_display_batch(lr_from_adjoint, 4),   # A_adjoint(LR)
        _to_display_batch(hr_A_then_adjoint, 4), # A_adjoint(A(HR))
        _to_display_batch(x_hat, 4),             # x_hat
    ]
    titles = ["LR", "HR", "A(HR)", "A_adjoint(LR)", "A_adjoint(A(HR))", "x_hat"]

    n_rows = min(4, y.shape[0])
    n_cols = len(samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for col, (imgs, title) in enumerate(zip(samples, titles)):
        axes[0, col].set_title(title)
        for row in range(n_rows):
            axes[row, col].imshow(imgs[row], cmap="gray")
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


# # # compute PSNR
# # # in_psnr = dinv.metric.PSNR()(x_gt, y).item()
# # out_psnr = dinv.metric.PSNR()(x_gt, x_hat).item()

# # dinv.utils.plot([x_gt, y, x_hat], ["Original", "Measurement\n Scale Factor " + str(scale_factor), "Finetuned reconstruction\n PSNR = {:.2f}dB".format(out_psnr)])


