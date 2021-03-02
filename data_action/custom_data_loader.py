import torch
import torchvision
import torchvision.transforms as transforms
from datasets.tinyimagenet_dataset import TinyImageNetDataset
from util.tiny_imagenet_datasource import TinyImageNetDataSource
from data_action.transformations import get_transforms

class CustomDataLoader():
    def __init__(self, dataset, device, batch_size,  test_split):
        self.dataset = dataset
        self.datasource = TinyImageNetDataSource(test_split) if self.dataset == 'TINY-IMAGENET' else None
        self.kwargs = {'num_workers': 2, 'pin_memory': True} if device=="cuda" else {}
        self.device = device
        self.batch_size = batch_size
        
    def load_training_data(self, transform_type):
        print(f'Loading training data. Dataset: {self.dataset}')
        trainloader = None
        if self.dataset == 'TINY-IMAGENET':
            trainloader = torch.utils.data.DataLoader(
                TinyImageNetDataset(datasource = self.datasource, train=True, transform=get_transforms(transform_type, self.datasource.mean, self.datasource.std, height=64, width=64)), 
                self.batch_size,
                shuffle=True,
                **self.kwargs)
        print('Training data loaded\n')
        return trainloader

    def load_testing_data(self):
        print('Loading testing data.')
        testloader = None
        if self.dataset == 'TINY-IMAGENET':
            testloader = torch.utils.data.DataLoader(
                TinyImageNetDataset(datasource = self.datasource, train=False, transform=get_transforms('pmda', self.datasource.mean, self.datasource.std)),
                self.batch_size,
                shuffle=False, 
                **self.kwargs)
        print('Test data loaded\n')
        return testloader
            