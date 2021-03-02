from torch.utils.data import Dataset
from util.tiny_imagenet_datasource import TinyImageNetDataSource


class TinyImageNetDataset(Dataset):
    
    def __init__(self, data_source, train = False, transform = None):  
        if not isinstance(data_source, TinyImageNetDataSource):
            print('Please send a vaid datasource instance')
            raise TypeError('{} is not an DataSource'.format(
                type(data_source).__name__))
            
        self.transform = transform
        self.dataset = None
        if train:
            self.dataset = data_source.get_trainset()
        else:
            self.dataset = data_source.get_testset()
          
            
    def __getitem__(self, index):
        image = self.dataset[index][0]
        y_label = self.dataset[index][1]
        
        if self.transform:
            image = self.transform(image)

        return image, y_label
