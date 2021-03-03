import torch
from torch.utils.data import Dataset
from util.tiny_imagenet_datasource import TinyImageNetDataSource


class TinyImageNetDataset(Dataset):
    
    def __init__(self, datasource, device, train = False, transform = None):  
        if not isinstance(datasource, TinyImageNetDataSource):
            print('Please send a vaid datasource instance')
            raise TypeError('{} is not an DataSource'.format(
                type(datasource).__name__))
            
        self.transform = transform
        self.device = device
        self.dataset = None
        if train:
            self.dataset = datasource.get_trainset()
        else:
            self.dataset = datasource.get_testset()
          
            
    def __getitem__(self, index):
        image = self.dataset[index][0]
        y_label = torch.tensor(self.dataset[index][1]).to(self.device)        
        if self.transform:
            image = self.transform(image)
        return image, y_label
    
    def __len__(self):
        return len(self.dataset)
