# .venv310/Scripts/python ram/mpi/finetune_mpi_supervised.py 

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ram.mpi.utils.mpiDataset import MpiDataset
from ram.mpi.utils.mpiSuperResPhysics import MPISuperResPhysics

import torch
import deepinv as dinv
from ram.models.ram import RAM, finetune_mpi
from torchvision import transforms
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

from ram.mpi.utils.loadData import get_train_data

bs = 25 # keep this one, do not change
lr = 0.00005 # 0.000025
step_num = 10e4 # 20e4
scale_factor = 2
file_dir = "ram/mpi/data/train/"
useAugmentation = False
max_iter = 200
visualize = True

snrThreshold = 5
n1 = n2 = 32

additional_info = ''
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
final_ckpt_path = f"finetune_mpi_ckp_scale{scale_factor}x_bs{bs}_iter{max_iter}_{timestamp}{additional_info}.pth.tar"

trainDS, copymax, copymin, epoch_nb = get_train_data(file_dir, scale_factor, bs, step_num, n1, n2, snrThreshold, useAugmentation, use_global_max_min=False)

# visualization of the first image in dataset
if visualize:
    img = trainDS.HR[0]
    if isinstance(img, torch.Tensor):
        x_true = img.float().unsqueeze(0).to(device)
    else:
        x_true = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )(img).unsqueeze(0).to(device)
    x_vis = x_true.clone()

    trainDS_bs = MpiDataset(
        trainDS.HR[0].clone(),
        trainDS.LR[0].clone(),
        maxhr=copymax[0],
        minhr=copymin[0],
        batch_size=bs,
        up=0.7,
        down=0.15
    )

    physics = MPISuperResPhysics(
        dataset=trainDS_bs,
        scale_factor=scale_factor
    )

    y_vis = physics(x_vis)
    img_list = [x_vis, y_vis]
    img_titles = ["Ground-Truth", "Measurement"]
    dinv.utils.plot(img_list, titles=img_titles)

model = RAM(device=device)

print(f"\nRunning RAM fine-tuning for scale x{scale_factor}...\n")

number_of_samples_to_be_trained = 200 # trainDS.LR.shape[0]
for i in range(0, number_of_samples_to_be_trained, bs):
    print(f"Data index: {i}/{number_of_samples_to_be_trained}")

    y = trainDS.LR[i:i+bs].to(device)
    x_gt = trainDS.HR[i:i+bs].to(device)

    trainDS_bs = MpiDataset(
        x_gt.clone(),
        y.clone(),
        maxhr=copymax[i:i+bs],
        minhr=copymin[i:i+bs],
        batch_size=bs,
        up=0.7,
        down=0.15
    )

    physics = MPISuperResPhysics(
        dataset=trainDS_bs,
        scale_factor=scale_factor
    )

    is_last_batch = (i + bs) >= number_of_samples_to_be_trained
    data = [x_gt, y]
    if not is_last_batch:
        model = finetune_mpi(model, data, physics, supervised=True, 
                            max_iter=max_iter, transform=None, lr=lr, 
                            early_stop=False, batch_size=bs, validation=data, # try when the validation is the next image
                            save_path=None, final_ckpt_path=None, is_last=False,)
    else:
        model = finetune_mpi(model, data, physics, supervised=True, 
                            max_iter=max_iter, transform=None, lr=lr, 
                            early_stop=False, batch_size=bs, validation=data, 
                            save_path=None,  # disable Trainer saves
                            final_ckpt_path=final_ckpt_path,  # write once
                            is_last=True) # try when the validation is the next image
    


# apply model
with torch.no_grad():
    out = model(y, physics=physics)
    zero_shot_psnr = dinv.metric.PSNR()(x_gt, out).mean()


# apply model
with torch.no_grad():
    out2 = model(y, physics=physics)
    finetuned_psnr = dinv.metric.PSNR()(x_gt, out2).mean()


print(f'PSNR zero-shot {zero_shot_psnr:.2f} dB')
print(f'PSNR finetuned {finetuned_psnr:.2f} dB')

# img_list.append(out2)
# img_titles.append("Finetuned")
# dinv.utils.plot(img_list, titles=img_titles)

img_list = [x_vis, y_vis, out[:1], out2[:1]]
img_titles = ["Ground-Truth", "Measurement", "Zero-Shot", "Finetuned"]
dinv.utils.plot(img_list, titles=img_titles)