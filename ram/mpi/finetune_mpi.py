# .venv310/Scripts/python ram/mpi/finetune_mpi.py

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
scale_factor = 2
file_dir = "ram/mpi/data/train/"
useAugmentation = False

snrThreshold = 5
n1 = n2 = 32
torch.cuda.set_device(useGPUno)

# trainLoader = loadMtxFromOpenMPI(file_dir, scale_factor, n1, n2, True, True)
# trainLoader.preprocessAndScaleSysMtx()
# print("local min max HR:", trainLoader.Hr_list[0][0][0][0])
# print("local min max LR:", trainLoader.Lr_list[0][0][0][0])
# print()

trainLoader = loadMtxFromOpenMPI(file_dir, scale_factor, n1, n2, True, True)
trainLoader.preprocessAndScaleMtxGlocally()
# by using global max min to normalize, the values became very similar to each other
# print("global min max HR:", trainLoader.Hr_list[0][0][0][0])
# print("global min max LR:", trainLoader.Lr_list[0][0][0][0])
# print()

totNoisePerElem = (2 * n1 * n2) ** (1/2)
print("Total Noise Per System Matrix: ",totNoisePerElem)

trainlr = torch.cat(tuple(trainLoader.Lr_list),0)
trainhr = torch.cat(tuple(trainLoader.Hr_list),0)
# trainBi = torch.cat(tuple(trainLoader.Bicubic_list), 0)
max_arr = np.reshape(trainLoader.mxList,(-1,1))
min_arr = np.reshape(trainLoader.mnList,(-1,1))
nsStdEst_arr = torch.cat(tuple(trainLoader.nsKestirimi_list), 0)

def denormalize(x,max,min,up,down):
    if (len(x.shape) - len(max.shape)) == 2:
        return ((x-down)/up)*(max[:,None,None]-min[:,None,None])+min[:,None,None]
    else:
        return ((x-down)/up)*(max[:,None,None,None]-min[:,None,None,None])+min[:,None,None,None]
    

train_hrDenorm = denormalize(trainhr, max_arr, min_arr, 0.7, 0.15)
sigPow = torch.sum(torch.sum(torch.sum(torch.abs(train_hrDenorm)**2, axis = 3), axis = 2), axis = 1).squeeze().sqrt()
print("Train Samples: {0}".format(trainlr.shape[0]))

snrEst_arr = sigPow / totNoisePerElem
selectedElems = (snrEst_arr > snrThreshold).squeeze()

trainlr = trainlr[selectedElems, :, :, :]
trainhr = trainhr[selectedElems, :, :, :]
max_arr = max_arr[selectedElems, :]
min_arr = min_arr[selectedElems, :]
nsStdEst_arr = nsStdEst_arr[selectedElems, :]
snrEst_arrCropped = snrEst_arr[selectedElems]

train_hrDenorm = denormalize(trainhr, max_arr, min_arr, 0.7, 0.15)
sigPows = torch.sum(torch.sum(torch.abs(train_hrDenorm)**2, axis = 3), axis = 2).squeeze().sqrt()
print("Train Samples after Filtering: {0}".format(trainlr.shape[0]))

ordercopy = np.linspace(0,trainlr.shape[0]-1,trainlr.shape[0],dtype=int)
np.random.seed(64)
np.random.shuffle(ordercopy)
trainhr = trainhr[ordercopy,:,:,:]
trainlr = trainlr[ordercopy,:,:,:]
max_arr = max_arr[ordercopy]
min_arr = min_arr[ordercopy]
nsStdEst_arr = nsStdEst_arr[ordercopy]
copylr = torch.clone(trainlr)
copyhr = torch.clone(trainhr)
copymax = torch.clone(torch.Tensor(max_arr))
copymin = torch.clone(torch.Tensor(min_arr))


## Add test data load here if needed ##


ordertrain = np.linspace(0,trainlr.shape[0]-1,trainlr.shape[0],dtype=int)
trainhr = copyhr
trainlr = copylr
maxhr = copymax.cuda()
minhr = copymin.cuda()

if (useAugmentation):
    trainhrAggr = torch.cat((trainhr, torch.flip(trainhr, [2]), torch.flip(trainhr, [3]), torch.flip(trainhr, [2, 3])) , dim = 0)
    trainlrAggr = torch.cat((trainlr, torch.flip(trainlr, [2]), torch.flip(trainlr, [3]), torch.flip(trainlr, [2, 3])) , dim = 0)
    maxhrAggr = torch.cat( (maxhr, maxhr, maxhr, maxhr), dim = 0 )
    minhrAggr = torch.cat( (minhr, minhr, minhr, minhr), dim = 0 )

    ordertrainAggr = np.linspace(0,trainlrAggr.shape[0]-1,trainlrAggr.shape[0],dtype=int)

batch_size = bs


if (useAugmentation):
    trainDS = MpiDataset(trainhrAggr, trainlrAggr, maxhrAggr, minhrAggr, batch_size, 0.7, 0.15, ordertrainAggr)
    trainDS.shuffleAll()
    train_size = np.ceil(trainlrAggr.shape[0]//batch_size) # from code
    epoch_nb = int(math.ceil(20e4 / train_size)) # from code
else:
    trainDS = MpiDataset(trainhr, trainlr, maxhr, minhr, batch_size, 0.7, 0.15, ordertrain)
    trainDS.shuffleAll()
    train_size = np.ceil(trainlr.shape[0]//batch_size) # from code
    epoch_nb = int(math.ceil(20e4 / train_size)) # from code

print("Num Epochs:",epoch_nb)


# model init
model_ram = RAM(device=device)

print(f"\nRunning RAM fine-tuning for scale x{scale_factor}...\n")
# currently the bs is 1
bs = 2048
inner_bs = 32
# for i in range(0, 5, bs): # used for visualizing 5 samples. set display_some_samples to True and comment all lines after that if block
for i in range(0, trainDS.LR.shape[0], bs):
    y = trainDS.LR[i:i+bs].to(device)   # [1, C, H, W]
    x_gt = trainDS.HR[i:i+bs].to(device)

    trainDS_bs = MpiDataset(
        x_gt.clone(),
        y.clone(),
        maxhr=copymax[i:i+bs],
        minhr=copymin[i:i+bs],
        batch_size=inner_bs,
        up=0.7,
        down=0.15
    )

    physics = MPISuperResPhysics(
        dataset=trainDS_bs,
        scale_factor=scale_factor,
        device=device
    )

    model = finetune(model_ram, y, physics, early_stop=False, transform=None, max_iter=100, noise_loss='noiseless', lr=1e-5, batch_size=inner_bs)

    # run inference
    with torch.no_grad():
        x_hat = model_ram(y, physics=physics)
    
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
            _to_display_batch(x_gt, 4),              # HR
            _to_display_batch(y, 4),                 # LR
            _to_display_batch(hr_from_A, 4),         # A(HR)
            _to_display_batch(lr_from_adjoint, 4),   # A_adjoint(LR)
            _to_display_batch(hr_A_then_adjoint, 4), # A_adjoint(A(HR))
            _to_display_batch(x_hat, 4),             # x_hat
        ]
        titles = ["HR", "LR", "A(HR)", "A_adjoint(LR)", "A_adjoint(A(HR))", "x_hat"]

        n_rows = min(4, y.shape[0])
        n_cols = len(samples)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]

        for col, (imgs, title) in enumerate(zip(samples, titles)):
            axes[0, col].set_title(title)
            for row in range(n_rows):
                axes[row, col].imshow(imgs[row], cmap="gray")
                # axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()

        break

    # # compute PSNR
    # # in_psnr = dinv.metric.PSNR()(x_gt, y).item()
    # out_psnr = dinv.metric.PSNR()(x_gt, x_hat).item()

    # dinv.utils.plot([x_gt, y, x_hat], ["Original", "Measurement\n Scale Factor " + str(scale_factor), "Finetuned reconstruction\n PSNR = {:.2f}dB".format(out_psnr)])


