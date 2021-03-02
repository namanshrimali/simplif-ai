import requests
from requests import get, HTTPError
from torch.utils.data import Dataset
import zipfile
import os

class TinyImageNetDataset(Dataset):
    download_directory = '../data'
    url = 'http://cs231n.stanford.edu'
    filename = 'tiny-imagenet-200.zip'
    
    def __init__(self, test_split, train_split, transform= None, train= True) -> None:        
        
        self.train_split = train_split/(train_split + test_split)
        self.test_split = 1 - train_split
        
        self.transform = transform
        
        if (not os.path.isdir(self.download_directory)) or (len(os.listdir(self.download_directory)) == 0):
            self._download_and_extract()
        else: 
            print('Files already downloaded and verified')
            
        _get_id_dictionary()
          
    def _download_and_extract(self):
        print(f'Downloading {self.url}/{self.filename} to {self.download_directory}/{self.filename}')
            
        try:
            resp = requests.get(f"{self.url}/{self.filename}", allow_redirects=True, stream=True)
        except HTTPError as http_error:
            print(f'HTTP error occurred: {http_error}')
        except Exception as exception:
            print(f'Other error occurred: {exception}')
        else:
            print("Dataset downloaded successfully")
            
        filename = f"{self.filename}"
        zfile = open(filename, 'wb')
        zfile.write(resp.content)
        zfile.close()
        
        print(f'Extracting {self.download_directory}/{filename} to {self.download_directory}')

        zipf = zipfile.ZipFile(filename, 'r')  
        zipf.extractall(self.download_directory)
        zipf.close()

        os.remove(filename)
    
    def _get_id_dictionary(self):
        id_dict = {}
        for i, line in enumerate(open(f"{self.download_directory}/wnids.txt", 'r')):
            id_dict[line.replace('\n', '')] = i
        print(id_dict)
        return id_dict
    