import torch
import torch.nn as nn
import torch.nn.functional as F

def test(model, device, test_loader, test_losses, test_accuracy, misclassified = []):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim= True)  # get the index of the max log-probability

            for focussed_data, prediction, actual in zip(data, pred, target):
                if prediction != actual:
                    misclassified.append([focussed_data, actual, prediction])


            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy.append(100. * correct / len(test_loader.dataset))
    
    return test_loss