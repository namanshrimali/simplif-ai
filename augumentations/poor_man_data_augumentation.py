import albumentations as A
from albumentations.pytorch import ToTensor

def get_poor_man_data_aug(mean, standard_deviation):
    return [
        A.Normalize(mean=mean, std=standard_deviation),
        ToTensor()
    ]
