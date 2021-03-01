from requests import get, HTTPError
import zipfile
import os

class GetDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        
        
        self._download_dataset()
          
    def _download_dataset(self):
        if self.dataset == 'TINY-IMAGENET':
            
            url = 'http://cs231n.stanford.edu'
            filename = 'tiny-imagenet-200.zip'
            
            try:
                resp = requests.get(f"{url}/{filename}")
            except HTTPError as http_error:
                print(f'HTTP error occurred: {http_err}')
            except Exception as exception:
                print(f'Other error occurred: {exception}')
            else:
                print("Dataset doanloaded successfully")
            
        
        