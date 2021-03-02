import os
import zipfile
from skimage import io
import torch
import requests
from requests import get, HTTPError
from util.mean_std_calc import calc_mean_std


class TinyImageNetDataSource():
    download_directory = '../data'
    url = 'http://cs231n.stanford.edu'
    sub_directory = 'tiny-imagenet-200'
    filename = f'{sub_directory}.zip'
    num_classes = 200
    image_per_class = 500
    
    def __init__(self, test_split):    
            if (not os.path.isdir(self.download_directory)) or (len(os.listdir(self.download_directory)) == 0):
                self._download_and_extract()
            else: 
                print('Files already downloaded and verified')
            
            dataset = self._get_data()
            
            dataset_len = len(dataset)
            test_data_len = int(test_split * dataset_len)
            train_data_len = dataset_len - test_data_len
            self.train_data, self.test_data = torch.utils.data.random_split(dataset, [train_data_len, test_data_len])
            self.mean, self.std = calc_mean_std(dataset)
            
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
        for i, line in enumerate(open(f"{self.download_directory}/{self.sub_directory}/wnids.txt", 'r')):
            id_dict[line.replace('\n', '')] = i
        return id_dict
    
    def _get_class_to_id_dict(self):
        id_dict = self._get_id_dictionary()
        all_classes = {}
        result = {}
        for _, line in enumerate(open(f"{self.download_directory}/{self.sub_directory}/words.txt", 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            result[value] = (key, all_classes[key])
        return result
    
    def _get_data(self):
        print('Starting to load data')
        dataset = []
    
        id_dict = self._get_id_dictionary()
        
        for (class_id, class_number) in id_dict.items():
            img_path = f'{self.download_directory}/{self.sub_directory}/train/{class_id}/images'
            
            for img_num in range(self.image_per_class):
                image_path = f"{img_path}/{class_id}_{img_num}.JPEG"
                dataset.append((io.imread(image_path), class_number))       
        print('Data loaded successfully')       
        return dataset
    
    def get_trainset(self):
        return self.train_data
    def get_testset(self):
        return self.test_data