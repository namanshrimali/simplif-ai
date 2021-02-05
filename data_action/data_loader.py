from data_action.transformations import *

class Data_Loader:
    def __init__(self, device, batch_size, dataset, mean, std, transform_type='pmda'):
        self.device = device
        self.batch_size = batch_size
        self.transform_type = transform_type
        self.dataset = dataset
        self.kwargs = {'num_workers': 2, 'pin_memory': True} if device=="cuda" else {}
        self.mean = mean
        self.std = std
    
    def load_training_data(self):
        print(f'Loading training data. Dataset: {self.dataset}')
        trainloader = None
        if self.dataset == 'CIFAR10':
            trainloader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=get_transforms(self.transform_type, self.mean, self.std)), 
                self.batch_size,
                shuffle=True,
                **self.kwargs)
        print('Training data loaded\n')
        return trainloader

    def load_testing_data(self):
        print('Loading testing data.')
        testloader = None
        if self.dataset == 'CIFAR10':
            testloader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=get_transforms('pmda', self.mean, self.std)),
                self.batch_size,
                shuffle=False, 
                **self.kwargs)
        print('Test data loaded\n')
        return testloader
