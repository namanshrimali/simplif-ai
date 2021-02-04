from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


def train(model, device, train_loader, train_loss, train_accuracy, optimizer):
    correct = 0
    processed = 0
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, position = 0, leave = True)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Predict
        output = model(data)
        

        loss = criterion(output, target)
        train_loss.append(loss)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)


        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
        train_accuracy.append(100*correct/processed)
    
    print('\Train set: Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))