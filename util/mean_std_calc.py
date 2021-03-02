import torch
import torchvision
import torchvision.transforms as transforms


def calc_mean_std(dataset):
    mean = 0
    std = 0
        
    for images in dataset:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataset)
    std /= len(dataset)
    mean = mean.numpy()
    std = std.numpy()
    print(f'Mean: {mean}, Std: {std}')
    return mean, std