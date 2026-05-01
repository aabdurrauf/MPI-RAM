# .venv310/Scripts/python ram/mpi/zero_shot.py --testFolder ram/mpi/data/test/ --saveOutFolder ram/mpi/mat_result/

# import libraries
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

import glob
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# from utils.mpiSuperResUtils import *
from utils.mpiSuperResPhysics import MPISuperResPhysics
from utils.mpiDataset import MpiDataset
from utils.loadData import *
import gc

import argparse

import torch.nn.functional as F
from ram.models.ram import RAM

# def A(x, scale_factor):
#     return F.avg_pool2d(x, kernel_size=scale_factor, stride=scale_factor)

# # def AT(y, scale_factor):
# #     return F.interpolate(y, scale_factor=scale_factor, mode='nearest')

# def AT(y, scale_factor):
#     y_up = F.interpolate(y, scale_factor=scale_factor, mode='nearest')
#     return y_up / (scale_factor ** 2)


visualize = False
device = "cuda"
# scale_factor, snrThreshold, useNoisyProjection, useGPUno, namePrefix, lrInpList, bsInpList, schedChoice, wdInpList

parser = argparse.ArgumentParser(description="TranSMS parameters")
parser.add_argument("--useGPUno", type=int, default=0,
                    help="Selected GPU no")
parser.add_argument("--useNoisyProjection", type=int, default=1,
                    help="0: Don't use noise projection, 1: use noise projection")
parser.add_argument("--bs", type=int, default=1024,
                    help="Batch Size")
parser.add_argument("--n1", type=int, default=32,
                    help="System Matrix Dimension")
parser.add_argument("--n2", type=int, default=32,
                    help="System Matrix Dimension")
parser.add_argument("--modelFolder", type=str, default="./outs/",
                    help="System Matrix Dimension")
parser.add_argument("--saveOutFolder", type=str, default="./newerTrial/",
                    help="System Matrix Dimension")
parser.add_argument("--testFolder", type=str, default="./test/",
                    help="System Matrix Dimension")
parser.add_argument("--interpolationMatrixPath", type=str, default="interpolaters.mat",
                    help="System Matrix Dimension")
parser.add_argument("--visualize", type=str, default=False,
                    help="Visualize LR vs HR")
opt = parser.parse_args()
print(opt)


namePrefix = "test_"


test_file_dir = opt.testFolder
readDir = opt.modelFolder
saveDir = opt.saveOutFolder
visualize = opt.visualize

def test_model_wbatch_DS(model, testDS):
    with torch.no_grad():
        model.eval()
        x = testDS.LR
        y_ground = testDS.HR
        y_model = torch.zeros_like(y_ground)
        iii = 0
        while(iii<y_model.shape[0]-(y_model.shape[0]%testDS.bs)):
            y_model[iii:iii+testDS.bs,:,:,:] = model(x[iii:iii+testDS.bs,:,:,:])
            iii += testDS.bs
        y_model[iii:,:,:,:] = model(x[iii:,:,:,:])
        return torch.norm(testDS.denormalize(y_model)-testDS.denormalize(y_ground))/torch.norm(testDS.denormalize(y_ground)), y_model


def inferenceFunction(opt):
    n1, n2 = opt.n1, opt.n2

    # Model Parameters
    useNoisyProjection = opt.useNoisyProjection
    bs = opt.bs
    
    # Select GPU
    useGPUno = opt.useGPUno
    torch.cuda.set_device(useGPUno)

    # regular downsampling experiments
    for scale_factor in [2, 4, 8]:
        testLoader = loadMtxFromOpenMPI(test_file_dir, scale_factor, n1, n2, True, True)
        testLoader.preprocessAndScaleSysMtx()

        totNoisePerElem = (2 * n1 * n2) ** (1/2)

        print("Total Noise Per normalized SM row: ", totNoisePerElem)
        
        test_lr = torch.cat(tuple(testLoader.Lr_list),0) # find out how to visualize this low resolution
        test_hr = torch.cat(tuple(testLoader.Hr_list),0) # and this high resolution SM using matplotlib.
        test_max_arr = np.reshape(testLoader.mxList,(-1,1))
        test_min_arr = np.reshape(testLoader.mnList,(-1,1))

        test_copylr = torch.clone(test_lr)
        test_copyhr = torch.clone(test_hr)
        test_copymax = torch.clone(torch.Tensor(test_max_arr))
        test_copymin = torch.clone(torch.Tensor(test_min_arr))

        # Debug: print first 10 samples of max/min normalization values
        n_debug = min(10, test_copymax.shape[0], test_copymin.shape[0])
        print(f"First {n_debug} test_copymax samples:", test_copymax[:n_debug].flatten().tolist())
        print(f"First {n_debug} test_copymin samples:", test_copymin[:n_debug].flatten().tolist())

        testDS = MpiDataset(test_copyhr, test_copylr, maxhr = test_copymax, minhr = test_copymin, batch_size = bs, up = 0.7, down = 0.15)
        totNoisePerElem = (2 * n1 * n2) ** (1/2)
        nsPwrProjection = totNoisePerElem / scale_factor ** 2

        # visualize test dataset using matplotlib
        # run using this: .venv310/Scripts/python ram/mpi/zero_shot.py --testFolder ram/mpi/data/test/
        if visualize and testDS.LR.shape[0] > 0 and testDS.HR.shape[0] > 0:
            sample_idx = 0
            lr_sample = testDS.LR[sample_idx].detach().cpu().numpy()
            hr_sample = testDS.HR[sample_idx].detach().cpu().numpy()

            lr_magnitude = np.sqrt(np.sum(np.square(lr_sample), axis=0)) if lr_sample.ndim == 3 else lr_sample
            hr_magnitude = np.sqrt(np.sum(np.square(hr_sample), axis=0)) if hr_sample.ndim == 3 else hr_sample

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].imshow(lr_magnitude, cmap="viridis")
            axes[0].set_title(f"LR sample (x{scale_factor})")
            axes[0].axis("off")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(hr_magnitude, cmap="viridis")
            axes[1].set_title("HR sample")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()

        # ---------------- RAM SETUP ----------------
        model_ram = RAM(device=device)
        # Loop over dataset
        print(f"\nRunning RAM zero-shot for scale x{scale_factor}...\n")

        ram_outputs = []
        ram_nrmse_list = []

        # EXAMINE AGAIN, DO WEE NEED TO PASS SAMPLE ONE BY ONE? CANT WE USE BATCH?
        # https://chatgpt.com/c/69e4a971-b800-83eb-95e1-4187913ff28e 
        
        # LEST TRY BATCH. 
        # RESULT: only batch size of 1 will work, 
        # otherwise will throw size mismatch error somewhere
        bs = 1
        for i in range(0, testDS.LR.shape[0], bs):
            # ---------------- Prepare single sample ---------------- 
            y = testDS.LR[i:i+bs].to(device)   # [1, C, H, W]
            x_gt = testDS.HR[i:i+bs].to(device)

            # VERY IMPORTANT: fix normalization per sample
            testDS_bs = MpiDataset(
                x_gt.clone(),
                y.clone(),
                maxhr=test_copymax[i:i+bs],
                minhr=test_copymin[i:i+bs],
                batch_size=bs,
                up=0.7,
                down=0.15
            )

            # ---------------- Physics ----------------
            physics = MPISuperResPhysics(
                dataset=testDS_bs,
                scale_factor=scale_factor,
                device=device
            )

            # ---------------- RAM inference ----------------
            with torch.no_grad():
                x_hat = model_ram(y, physics=physics)

            ram_outputs.append(x_hat.detach().cpu())

            # ---------------- Compute nRMSE ----------------
            x_hat_denorm = testDS_bs.denormalize(x_hat)
            x_gt_denorm = testDS_bs.denormalize(x_gt)

            # ---------------- Visualization (PUT HERE) ----------------
            if i < 100 and i%10==0:
                plt.figure(figsize=(10,3))

                plt.subplot(1,3,1)
                plt.imshow(torch.norm(x_gt_denorm[0], dim=0).cpu(), cmap='viridis')
                plt.title("GT")

                plt.subplot(1,3,2)
                plt.imshow(torch.norm(y[0], dim=0).cpu(), cmap='viridis')
                plt.title("LR")

                plt.subplot(1,3,3)
                plt.imshow(torch.norm(x_hat_denorm[0], dim=0).cpu(), cmap='viridis')
                plt.title("RAM")

                plt.tight_layout()
                plt.show()

                plt.figure()
                plt.imshow(torch.norm(x_hat_denorm[0] - x_gt_denorm[0], dim=0).cpu(), cmap='hot')
                plt.title("Error (RAM - GT)")
                plt.colorbar()
                plt.show()

            err = torch.norm(x_hat_denorm - x_gt_denorm)
            denom = torch.norm(x_gt_denorm)

            nrmse = (err / denom).item()
            ram_nrmse_list.append(nrmse)

            if i % 50 == 0:
                print(f"[RAM] Sample {i}/{testDS.LR.shape[0]} nRMSE: {nrmse:.6f}")

        # ---------------- Final results ----------------
        ram_outputs = torch.cat(ram_outputs, dim=0)
        mean_nrmse = np.mean(ram_nrmse_list)

        print("\n==============================")
        print(f"RAM Mean nRMSE (scale x{scale_factor}): {mean_nrmse:.6f}")
        print("==============================\n")

        # ---------------- Save results ----------------
        savemat(
            saveDir + f"RAM_x{scale_factor}_results.mat",
            {
                'x_hat': ram_outputs.numpy(),
                'nrmse_list': np.array(ram_nrmse_list),
                'mean_nrmse': mean_nrmse
            }
        )



###############################################################################
        # physics = MPISuperResPhysics(dataset, scale_factor=2, device=device)
        # model = RAM(device=device)
        
        # for file in sorted(os.listdir(readDir)):
        #     with torch.no_grad():
        #         if (file[-3:] == ".pt"):
        #             if ("{0}x".format(scale_factor) in file):
        #                 useNoisyProjection = 0

        #                 if ("CvTRDN" in file):
        #                     useNoisyProjection = 1
        #                     if (scale_factor == 2):
        #                         config={'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 24, 'scaleFactor': 2, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 5, 'rdn_growth_rate': 6, 'img_size1': n1 // 2, 'img_size2': n2 // 2, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}
        #                     elif (scale_factor == 4):
        #                         config = {'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 24, 'scaleFactor': 4, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 8, 'rdn_growth_rate': 6, 'img_size1': n1 // 4, 'img_size2': n2 // 4, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}
        #                     elif (scale_factor == 8):
        #                         config = {'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 64, 'scaleFactor': 8, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 9, 'rdn_growth_rate': 6, 'img_size1': n1 // 8, 'img_size2': n2 // 8, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}

        #                     model = par_cvt_rdnDualNonSq(config).cuda()
        #                 else:
        #                     continue
        #                 model.eval()
        #                 for para in model.parameters():
        #                     para.requires_grad = False
        #                 print("Processing: ",file)

        #                 model.load_state_dict(torch.load(readDir +file,map_location=torch.device('cuda:{0}'.format(useGPUno))))
        #                 nsPwrProjection = totNoisePerElem / scale_factor ** 2

        #                 gc.collect()
        #                 torch.cuda.empty_cache()

    #                     temp_test_nrmseDivider = float(torch.norm(testDS.denormalize(testDS.HR)))
    #                     temp_test_nrmse, y_model_out = test_model_wbatch_DS(model, testDS)
    #                     oldBs = testDS.bs
    #                     testDS.bs = testDS.LR.shape[0]
    #                     nsEpsTest = testDS.noisePowerNormalize(nsPwrProjection, 0).reshape(-1, 1)

    #                     y_model_NsPrj = torch.zeros_like(testDS.HR)
    #                     iii = 0
    #                     while(iii<y_model_NsPrj.shape[0]-(y_model_NsPrj.shape[0]%bs)):
    #                         y_model_NsPrj[iii:iii+bs] = projectToNoiseLevel(y_model_out[iii:iii+bs], testDS.LR[iii:iii+bs], nsEpsTest[iii:iii+bs], int(scale_factor), int(scale_factor))
    #                         iii += bs
    #                     y_model_NsPrj[iii:] = projectToNoiseLevel(y_model_out[iii:], testDS.LR[iii:], nsEpsTest[iii:], int(scale_factor), int(scale_factor))

    #                     temp_test_nrmse_new = float(torch.norm(testDS.noisePowerDeNormalize(y_model_NsPrj - testDS.HR, 0)))
    #                     testDS.bs = oldBs

    #                     temp_test_nrmse = temp_test_nrmse_new / temp_test_nrmseDivider
    #                     print("nRMSE for HR measurement: {0:.4f}".format(float((test_copyhr.shape[0]/2 * 32*32)**(1/2) / temp_test_nrmseDivider)))

        #                 torch.cuda.empty_cache()
        #                 gc.collect()
        #                 torch.cuda.empty_cache()

        #                 print("estimated nRMSE for all unfiltered rows: ", float(temp_test_nrmse))
        #                 savemat(saveDir + file[:-3].replace(":","_") +"_save_snr0.mat" ,{'y_model_NsPrj':testDS.denormalize(y_model_NsPrj).cpu().detach().numpy()})
    
if __name__ == "__main__":
    inferenceFunction(opt)

