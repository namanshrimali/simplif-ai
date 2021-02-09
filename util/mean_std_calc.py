import torch
import torchvision
import torchvision.transforms as transforms

def calc_mean_std(dataset):
    if dataset == 'CIFAR10':
        
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root='../data',
            download=True,
            transform= transforms.Compose([
                transforms.ToTensor()
                ])), 
            batch_size=10,
            shuffle=False,
            num_workers=0
        )
        
        mean = 0
        std = 0
        
        for images, _ in loader:
            batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        mean /= len(loader.dataset)
        std /= len(loader.dataset)
        mean = mean.numpy()
        std = std.numpy()
        print(f'Mean: {mean}, Std: {std}')
        return mean, std