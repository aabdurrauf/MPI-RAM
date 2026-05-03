import deepinv as dinv
import torch.nn.functional as F

# One way to check if the physics of operator A and A_adjoint is true is by visualizing it. 
# If after doing A and then A_adjoint the result is similar to the input, then it is ok, 
# otherwise, check the operator again.
# TODO: visualize it. DONE and the physics is correclty implemented.
class MPISuperResPhysics(dinv.physics.Physics):
    def __init__(self, dataset, scale_factor):
        super().__init__()
        self.dataset = dataset
        self.s = scale_factor

    def A(self, x):
        # x: normalized HR

        # 1. denormalize
        x_denorm = self.dataset.denormalize(x)

        # 2. forward (your downsampling)
        y = F.avg_pool2d(x_denorm, kernel_size=self.s, stride=self.s)

        # 3. normalize back
        y_norm = self.dataset.normalize(y)

        return y_norm

    def A_adjoint(self, y):
        # y: normalized LR

        # 1. denormalize
        y_denorm = self.dataset.denormalize(y)

        # 2. upsample (adjoint)
        x = F.interpolate(y_denorm, scale_factor=self.s, mode='nearest')

        # 3. scale correction
        x = x / (self.s ** 2)

        # 4. normalize back
        x_norm = self.dataset.normalize(x)

        return x_norm