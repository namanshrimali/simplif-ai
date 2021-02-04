import torch

def find_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}\n')
    return device