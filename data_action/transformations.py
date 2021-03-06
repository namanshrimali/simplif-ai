import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from augumentations.middle_man_data_augumentation import *
from augumentations.poor_man_data_augumentation import *
from augumentations.commonly_used_augumentation import *
import numpy as np

# requires installation of albumentations
def get_transforms(transform_type, mean, standard_deviation, height=32, width=32):
    if transform_type == 'pmda':
        aug = A.Compose(
            get_poor_man_data_aug(mean, standard_deviation)
            )
        return lambda img:aug(image=np.array(img))["image"]

    elif transform_type == 'mmda':
        aug = A.Compose(
            get_middle_man_data_aug(mean, standard_deviation, height=height, width=width)
            )
        return lambda img:aug(image=np.array(img))["image"]
    elif transform_type == 'common':
        aug = A.Compose(
            get_commonly_used_augumentation(mean, standard_deviation)
        )
        return lambda img:aug(image=np.array(img))["image"]

