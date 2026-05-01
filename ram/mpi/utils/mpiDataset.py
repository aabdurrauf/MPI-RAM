# import libraries
import numpy as np

class MpiDataset:
    def __init__(self, HRimages, LRimages, maxhr, minhr, batch_size, up = 0.7, down = 0.15, ordertrain = 0):
        self.HR = HRimages.cuda()
        self.LR = LRimages.cuda()
        self.bs = batch_size
        self.max = maxhr.cuda()
        self.min = minhr.cuda()
        self.up = up
        self.down = down
        self.ordertrain = ordertrain
    
    def shuffleAll(self):
        np.random.shuffle(self.ordertrain)
        self.LR = self.LR[self.ordertrain,:,:,:]
        self.HR = self.HR[self.ordertrain,:,:,:]
        self.max = self.max[self.ordertrain,:]
        self.min = self.min[self.ordertrain,:]
        
    def normalize(self, x):
        # return (x - self.min[:,None,None])/(self.max[:,None,None]-self.min[:,None,None])*self.up + self.down
        # return (x - self.min[:self.bs,None,None])/(self.max[:self.bs,None,None]-self.min[:self.bs,None,None])*self.up + self.down
        return (x - self.min[:x.shape[0],None,None])/(self.max[:x.shape[0],None,None]-self.min[:x.shape[0],None,None])*self.up + self.down
    
    def denormalize(self, x):
        # return ((x-self.down)/self.up)*(self.max[:,None,None]-self.min[:,None,None])+self.min[:,None,None]
        # return ((x-self.down)/self.up)*(self.max[:self.bs,None,None]-self.min[:self.bs,None,None])+self.min[:self.bs,None,None]
        return ((x-self.down)/self.up)*(self.max[:x.shape[0],None,None]-self.min[:x.shape[0],None,None])+self.min[:x.shape[0],None,None]
    
    def noisePowerNormalizeWObatch(self, x):
        return (x)/(self.max[:,None,None]-self.min[:,None,None])*self.up
        
    def noisePowerDeNormalizeWObatch(self, x):
        return (x)*(self.max[:,None,None]-self.min[:,None,None])/self.up
    
    def noisePowerNormalizelize(self, x, batch_start_idx):
        return (x)/(self.max[batch_start_idx:batch_start_idx+self.bs,None,None]-self.min[batch_start_idx:batch_start_idx+self.bs,None,None])*self.up
    
    def noisePowerDeNormalize(self, x, batch_start_idx):
        return (x)*(self.max[batch_start_idx:batch_start_idx+self.bs,None,None]-self.min[batch_start_idx:batch_start_idx+self.bs,None,None])/self.up
  