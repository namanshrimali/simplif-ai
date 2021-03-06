import albumentations as A
from albumentations.pytorch import ToTensor

def get_middle_man_data_aug(mean, standard_deviation, height, width):
    return [
        A.Normalize(mean=mean, std=standard_deviation, always_apply=True, p=1.0),
        A.PadIfNeeded(min_height=height+8, min_width=width+8),
        A.RandomCrop(height, width, always_apply=True, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.GridDistortion (num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
        A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5),
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=mean, always_apply=False, p=0.5),
        ToTensor()
    ]
