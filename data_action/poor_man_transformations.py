import torch
import torchvision
import torchvision.transforms as transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1, 1, 1)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])