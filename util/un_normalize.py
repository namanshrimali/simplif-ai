
class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    
    '''
        UnNormalizes an image given its mean and standard deviation
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
    '''
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
        