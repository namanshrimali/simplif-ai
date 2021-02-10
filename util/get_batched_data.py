import torch

def get_misclassified_as_batched_data(misclassified_data, batch_size):
    '''Takes in the list data and outputs data, targets and preds'''
    in_imgs = []
    in_preds = []
    in_targets = []
        
    for index, i in enumerate(misclassified_data):
        if index < batch_size:
            in_imgs.append(i[0])
            in_targets.append(i[1])
            in_preds.append(i[2])
        else:
            break

    return torch.stack(in_imgs), torch.stack(in_targets), torch.stack(in_preds)