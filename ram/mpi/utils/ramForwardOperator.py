import torch.nn.functional as F

class RAMForwardOperator:
    def __init__(self, dataset, scale_factor):
        self.dataset = dataset
        self.s = scale_factor

    def A(self, x):
        # x is NORMALIZED HR

        # 1. convert to real physical values
        x_denorm = self.dataset.denormalize(x) # i dont think the x is already normalized. idk

        # 2. apply your real forward model (downsampling)
        y = F.avg_pool2d(x, kernel_size=self.s, stride=self.s)

        # 3. convert back to normalized space
        y_norm = self.dataset.normalize(y)

        return y_norm

    def AT(self, y):
        # y is NORMALIZED LR

        # 1. convert to real physical values
        y_denorm = self.dataset.denormalize(y)

        # 2. backprojection (upsample)
        x = F.interpolate(y_denorm, scale_factor=self.s, mode='nearest')

        # 3. match scaling of forward
        x = x / (self.s ** 2)

        # 4. normalize again
        x_norm = self.dataset.normalize(x)

        return x_norm