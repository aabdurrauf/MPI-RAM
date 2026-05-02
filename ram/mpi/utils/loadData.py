# import libraries
import math

import torch
import numpy as np
from scipy.io import loadmat
import os
from ram.mpi.utils.mpiDataset import MpiDataset
from ram.mpi.utils.mpiSuperResUtils import downsampleImageNP, interpImage, normalize


class loadMtxFromOpenMPI:
    def __init__(self, file_dir, scale_factor, n1, n2, useBoxDownsampling = True, useTwoChannelInput = False):
        self.file_dir = file_dir
        self.scale_factor = scale_factor
        self.n1 = n1
        self.n2 = n2
        
        self.HrUnsc_list = list()
        self.LrUnsc_list = list()
        self.nsKestirimiUnsc_list = list()
        self.snrUnsc_list = list()
        self.BiCubicUnsc_list = list()
        LrUnsc_list = list()
        BiCubicUnsc_list = list()
        
        self.useTwoChannelInput = useTwoChannelInput
        self.n1Down = self.n1 // scale_factor
        self.n2Down = self.n2 // scale_factor
        
                    
        for file in sorted(os.listdir(file_dir+"/")):
            if (file[-4:] == ".mat"):
                readImg = loadmat(file_dir+"/"+file)
                hrMtx = readImg['myMtx']
                nsMtx1 = readImg['estNsKestirimi']
                snrMtx = readImg['snrList']
                
                numFreqs = nsMtx1.shape[0]
                
                numZslice = hrMtx.shape[0] // numFreqs
                
                hrMtx = hrMtx.reshape(numFreqs, numZslice, n1, n2)
                nsMtx1 = nsMtx1.repeat(numZslice,axis = 1)
                snrMtx = snrMtx.repeat(numZslice,axis = 1)
        
                self.HrUnsc_list.append(hrMtx.reshape(numZslice * numFreqs, n1, n2)) # Bütün freq'ler
            
                if (useBoxDownsampling):
                    lrMtx = downsampleImageNP(hrMtx.reshape(-1, n1, n2), scale_factor, scale_factor).reshape(numZslice * numFreqs, self.n1Down, self.n2Down) / scale_factor ** 2
                else:
                    lrMtx = (hrMtx.reshape(-1, n1, n2)[:,0::scale_factor, 0::scale_factor]).reshape(numZslice * numFreqs, self.n1Down, self.n2Down)
                self.LrUnsc_list.append(lrMtx)
                
                self.nsKestirimiUnsc_list.append(nsMtx1.reshape(-1))
                self.snrUnsc_list.append(snrMtx.reshape(-1))
                
        self.newSize = (-1, self.n1, self.n2)
        self.newSizeLR = (-1, self.n1Down, self.n2Down)
        self.how_many_data_file = len(self.HrUnsc_list)
        
        self.Hr_list = list()
        self.Lr_list = list()
        self.nsKestirimi_list = list()
        self.Bicubic_list = list()
    
    
    def preprocessAndScaleSysMtx(self):
        self.mxList = np.zeros((0))
        self.mnList = np.zeros((0))

        for myReadImNum in range(self.how_many_data_file):
            HRIm = self.HrUnsc_list[myReadImNum]
            LRIm = self.LrUnsc_list[myReadImNum]
            nsIm = self.nsKestirimiUnsc_list[myReadImNum]
            
            if (self.useTwoChannelInput):
                _HRList = np.concatenate((HRIm.real.reshape(-1, 1, self.n1, self.n2), HRIm.imag.reshape(-1, 1, self.n1, self.n2)), axis = 1)
                _LRList = np.concatenate((LRIm.real.reshape(-1, 1, self.n1Down, self.n2Down), LRIm.imag.reshape(-1, 1, self.n1Down, self.n2Down)), axis = 1)
                numCh = 2
            else:
                _HRList = np.concatenate((HRIm.real.reshape(self.newSize), HRIm.imag.reshape(self.newSize)))
                _LRList = np.concatenate((LRIm.real.reshape(self.newSizeLR), LRIm.imag.reshape(self.newSizeLR)))
                numCh = 1

            mx = _LRList.reshape(-1, numCh * self.n1Down*self.n2Down).max(axis = 1)
            mn = _LRList.reshape(-1, numCh * self.n1Down*self.n2Down).min(axis = 1)

            _HRListSc = normalize(_HRList,mx,mn,0.7,0.15)
            _LRListSc = normalize(_LRList,mx,mn,0.7,0.15)
            _nsKestirimiSc = nsIm.repeat(2 // numCh, axis = 0) / (mx - mn) * 0.7
            
            self.mxList = np.concatenate((self.mxList, mx), axis = 0)
            self.mnList = np.concatenate((self.mnList, mn), axis = 0)
            
            if (self.useTwoChannelInput):
                self.Hr_list.append((torch.as_tensor(_HRListSc,dtype=torch.double)).float())
                self.Lr_list.append((torch.as_tensor(_LRListSc,dtype=torch.double)).float())
            else:
                self.Hr_list.append(torch.unsqueeze(torch.as_tensor(_HRListSc,dtype=torch.double), 1).float())
                self.Lr_list.append(torch.unsqueeze(torch.as_tensor(_LRListSc,dtype=torch.double), 1).float())
            
            self.nsKestirimi_list.append(torch.unsqueeze(torch.as_tensor(_nsKestirimiSc,dtype=torch.double), 1).float())
    
    def preprocessAndScaleMtxGlocally(self):
        """
        Preprocess and scale the low-resolution and high-resolution MRI data globally.
        This method builds normalized tensor views from the stored unscaled complex-valued
        data and appends the processed results to the instance lists used for training or
        inference.
        The processing steps are:
        1. Initialize `mxList` and `mnList` as empty arrays for storing per-sample scaling
            bounds.
        2. Define a helper function that converts complex-valued images into either:
            - a two-channel representation with separate real and imaginary channels, or
            - a flattened single-channel representation, depending on `useTwoChannelInput`.
        3. Scan all low-resolution inputs to compute a single global maximum and minimum
            value across the entire dataset.
        4. Store those global bounds in `self.global_mx` and `self.global_mn`.
        5. For each data file:
            - build the high-resolution and low-resolution views,
            - create per-sample arrays filled with the global max and min,
            - normalize the HR and LR views using the shared global bounds,
            - scale the noise-estimation target accordingly,
            - append the scaling bounds to `mxList` and `mnList`,
            - convert the normalized arrays into PyTorch tensors,
            - add a channel dimension when single-channel input is used,
            - append the tensors to `Hr_list`, `Lr_list`, and `nsKestirimi_list`.
        Side Effects:
             - Populates `self.global_mx` and `self.global_mn`.
             - Extends `self.mxList` and `self.mnList`.
             - Appends normalized tensors to `self.Hr_list`, `self.Lr_list`,
                and `self.nsKestirimi_list`.
        Notes:
             - Normalization is performed with dataset-wide global min/max values rather
                than per-sample values.
             - The helper function uses the instance attributes `n1`, `n2`, `n1Down`,
                `n2Down`, `newSize`, `newSizeLR`, and `useTwoChannelInput`.
        """
        self.mxList = np.zeros((0))
        self.mnList = np.zeros((0))

        def _build_views(hr_im, lr_im):
            if self.useTwoChannelInput:
                hr_view = np.concatenate((
                    hr_im.real.reshape(-1, 1, self.n1, self.n2),
                    hr_im.imag.reshape(-1, 1, self.n1, self.n2)
                    ),
                    axis=1
                )
                lr_view = np.concatenate((
                    lr_im.real.reshape(-1, 1, self.n1Down, self.n2Down),
                    lr_im.imag.reshape(-1, 1, self.n1Down, self.n2Down)),
                    axis=1
                )
                num_ch = 2
            else:
                hr_view = np.concatenate((
                    hr_im.real.reshape(self.newSize),
                    hr_im.imag.reshape(self.newSize))
                )
                lr_view = np.concatenate((
                    lr_im.real.reshape(self.newSizeLR),
                    lr_im.imag.reshape(self.newSizeLR))
                )
                num_ch = 1
            return hr_view, lr_view, num_ch

        global_mx = -np.inf
        global_mn = np.inf

        for i in range(self.how_many_data_file):
            hr_im = self.HrUnsc_list[i]
            lr_im = self.LrUnsc_list[i]
            _, lr_view, _ = _build_views(hr_im, lr_im)
            global_mx = max(global_mx, lr_view.max())
            global_mn = min(global_mn, lr_view.min())

        self.global_mx = global_mx
        self.global_mn = global_mn

        print("Debug - global max:", self.global_mx, "- global min:", self.global_mn)

        for i in range(self.how_many_data_file):
            hr_im = self.HrUnsc_list[i]
            lr_im = self.LrUnsc_list[i]
            ns_im = self.nsKestirimiUnsc_list[i]

            hr_view, lr_view, num_ch = _build_views(hr_im, lr_im)
            mx = np.full((lr_view.reshape(-1, num_ch * self.n1Down * self.n2Down).shape[0],), global_mx)
            mn = np.full((lr_view.reshape(-1, num_ch * self.n1Down * self.n2Down).shape[0],), global_mn)

            hr_sc = normalize(hr_view, mx, mn, 0.7, 0.15)
            lr_sc = normalize(lr_view, mx, mn, 0.7, 0.15)
            ns_sc = ns_im.repeat(2 // num_ch, axis=0) / (mx - mn) * 0.7

            self.mxList = np.concatenate((self.mxList, mx), axis=0)
            self.mnList = np.concatenate((self.mnList, mn), axis=0)

            if self.useTwoChannelInput:
                self.Hr_list.append(torch.as_tensor(hr_sc, dtype=torch.double).float())
                self.Lr_list.append(torch.as_tensor(lr_sc, dtype=torch.double).float())
            else:
                self.Hr_list.append(torch.unsqueeze(torch.as_tensor(hr_sc, dtype=torch.double), 1).float())
                self.Lr_list.append(torch.unsqueeze(torch.as_tensor(lr_sc, dtype=torch.double), 1).float())

            self.nsKestirimi_list.append(torch.unsqueeze(torch.as_tensor(ns_sc, dtype=torch.double), 1).float())
            
    def getBicubicInterpolation(self):
        n1 = self.n1
        n2 = self.n2
        for myReadImNum in range(self.how_many_data_file):
            LRIm = self.Lr_list[myReadImNum].numpy()
            numData = LRIm.shape[0]
            
            dnmInterp = [torch.from_numpy(interpImage(LRIm[i,0], n1, n2)) for i in range(numData)]
            dnmInterp2 = torch.cat(tuple(dnmInterp),0).reshape(numData, 1, n1, n2)
            
            if (self.useTwoChannelInput):
                dnmInterp = [torch.from_numpy(interpImage(LRIm[i,1], n1, n2)) for i in range(numData)]
                dnmInterp3 = torch.cat(tuple(dnmInterp),0).reshape(numData, 1, n1, n2)
                dnmInterp2 = torch.cat((dnmInterp2, dnmInterp3), 1)
                
            self.Bicubic_list.append(dnmInterp2)

def get_train_data(file_dir, scale_factor, batch_size, step_num, n1=32, n2=32, snrThreshold=5, useAugmentation=False, use_global_max_min=True):
    trainLoader = loadMtxFromOpenMPI(file_dir, scale_factor, n1, n2, True, True)
    if use_global_max_min:
        trainLoader.preprocessAndScaleMtxGlocally()
    else:
        trainLoader.preprocessAndScaleSysMtx()
    # by using global max min to normalize, the values became very similar to each other
    # print("global min max HR:", trainLoader.Hr_list[0][0][0][0])
    # print("global min max LR:", trainLoader.Lr_list[0][0][0][0])

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

    if (useAugmentation):
        trainDS = MpiDataset(trainhrAggr, trainlrAggr, maxhrAggr, minhrAggr, batch_size, 0.7, 0.15, ordertrainAggr)
        trainDS.shuffleAll()
        train_size = np.ceil(trainlrAggr.shape[0]//batch_size) # from code
        epoch_nb = int(math.ceil(step_num / train_size)) # from code
    else:
        trainDS = MpiDataset(trainhr, trainlr, maxhr, minhr, batch_size, 0.7, 0.15, ordertrain)
        trainDS.shuffleAll()
        train_size = np.ceil(trainlr.shape[0]//batch_size) # from code
        epoch_nb = int(math.ceil(step_num / train_size)) # from code

    print("Num Epochs:",epoch_nb)

    return trainDS, copymax, copymin, epoch_nb

def get_eval_data(file_dir, scale_factor, batch_size, n1=32, n2=32, snrThreshold=5, use_global_max_min=True):
    evalLoader = loadMtxFromOpenMPI(file_dir, scale_factor, n1, n2, True, True)
    
    if use_global_max_min:
        evalLoader.preprocessAndScaleMtxGlocally()
    else:
        evalLoader.preprocessAndScaleSysMtx()

    totNoisePerElem = (2 * n1 * n2) ** (1/2)
    print("Total Noise Per System Matrix: ",totNoisePerElem)

    evallr = torch.cat(tuple(evalLoader.Lr_list),0)
    evalhr = torch.cat(tuple(evalLoader.Hr_list),0)
    max_arr = np.reshape(evalLoader.mxList,(-1,1))
    min_arr = np.reshape(evalLoader.mnList,(-1,1))
    nsStdEst_arr = torch.cat(tuple(evalLoader.nsKestirimi_list), 0)

    def denormalize(x,max,min,up,down):
        if (len(x.shape) - len(max.shape)) == 2:
            return ((x-down)/up)*(max[:,None,None]-min[:,None,None])+min[:,None,None]
        else:
            return ((x-down)/up)*(max[:,None,None,None]-min[:,None,None,None])+min[:,None,None,None]
        

    eval_hrDenorm = denormalize(evalhr, max_arr, min_arr, 0.7, 0.15)
    sigPow = torch.sum(torch.sum(torch.sum(torch.abs(eval_hrDenorm)**2, axis = 3), axis = 2), axis = 1).squeeze().sqrt()
    print("Evaluation Samples: {0}".format(evallr.shape[0]))

    snrEst_arr = sigPow / totNoisePerElem
    selectedElems = (snrEst_arr > snrThreshold).squeeze()

    evallr = evallr[selectedElems, :, :, :]
    evalhr = evalhr[selectedElems, :, :, :]
    max_arr = max_arr[selectedElems, :]
    min_arr = min_arr[selectedElems, :]
    nsStdEst_arr = nsStdEst_arr[selectedElems, :]
    snrEst_arrCropped = snrEst_arr[selectedElems]

    eval_hrDenorm = denormalize(evalhr, max_arr, min_arr, 0.7, 0.15)
    sigPows = torch.sum(torch.sum(torch.abs(eval_hrDenorm)**2, axis = 3), axis = 2).squeeze().sqrt()
    print("Evaluation Samples after Filtering: {0}".format(evallr.shape[0]))

    ordercopy = np.linspace(0,evallr.shape[0]-1,evallr.shape[0],dtype=int)
    np.random.seed(64)
    np.random.shuffle(ordercopy)
    evalhr = evalhr[ordercopy,:,:,:]
    evallr = evallr[ordercopy,:,:,:]
    max_arr = max_arr[ordercopy]
    min_arr = min_arr[ordercopy]
    nsStdEst_arr = nsStdEst_arr[ordercopy]
    copylr = torch.clone(evallr)
    copyhr = torch.clone(evalhr)
    copymax = torch.clone(torch.Tensor(max_arr))
    copymin = torch.clone(torch.Tensor(min_arr))

    ordereval = np.linspace(0,evallr.shape[0]-1,evallr.shape[0],dtype=int)
    evalhr = copyhr
    evallr = copylr
    maxhr = copymax.cuda()
    minhr = copymin.cuda()

    
    evalDS = MpiDataset(evalhr, evallr, maxhr, minhr, batch_size, 0.7, 0.15, ordereval)
    evalDS.shuffleAll()

    return evalDS, copymax, copymin
