import requests
from requests import get, HTTPError
import zipfile
import os

class GetDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        self.download_directory = '../data'
        
        if (not os.path.isdir(self.download_directory)) or (len(os.listdir(self.download_directory)) == 0):
            self._download_dataset()
        else: 
            print('Dataset already downloaded')
          
    def _download_dataset(self):
        url = None
        filename = None
        if self.dataset == 'TINY-IMAGENET':
            url = 'http://cs231n.stanford.edu'
            filename = 'tiny-imagenet-200.zip'
            
        try:
            resp = requests.get(f"{url}/{filename}", allow_redirects=True, stream=True)
        except HTTPError as http_error:
            print(f'HTTP error occurred: {http_error}')
        except Exception as exception:
            print(f'Other error occurred: {exception}')
        else:
            print("Dataset downloaded successfully")
            
        filename = f"{filename}"
        zfile = open(filename, 'wb')
        zfile.write(resp.content)
        zfile.close()

        zipf = zipfile.ZipFile(filename, 'r')  
        zipf.extractall(self.download_directory)
        zipf.close()

        os.remove(filename)
            
        
        