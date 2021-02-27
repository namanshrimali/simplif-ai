import albumentations as A
from albumentations.pytorch import ToTensor

def get_commonly_used_augumentation(mean, standard_deviation):
    return [
        A.Normalize(mean=mean, std=standard_deviation, always_apply=True, p=1.0),
        A.PadIfNeeded(min_height=40, min_width=40),
        A.RandomCrop(32, 32, always_apply=True, p=1.0),
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=mean, always_apply=True, p=1),
        ToTensor()
    ]
