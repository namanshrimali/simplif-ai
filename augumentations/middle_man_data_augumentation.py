import albumentations as A
from albumentations.pytorch import ToTensor

def get_middle_man_data_aug(mean, standard_deviation):
    return [
        A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5),
        A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=mean, always_apply=False, p=0.5),
        A.Normalize(mean=mean, std=standard_deviation),
        ToTensor()       
    ]
