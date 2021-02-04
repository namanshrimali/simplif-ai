import matplotlib.pyplot as plt
import torchvision
import numpy as np


def plot_images(data_iterable, classes, total= 5):

    images, actual, *predicted = data_iterable.__next__()
    images = images/2 + 0.5 #un-normalize
    plt.figure(figsize=(30, 30))

    if predicted == []:
        for image_num in range(total):
            plt.subplot(6, 6, image_num+1)    
            plt.imshow(np.transpose(images[image_num], (1, 2, 0)))
            plt.title(classes[actual[image_num]])
    else:
        for image_num in range(total):
            plt.subplot(5, 5, image_num+1)
            plt.imshow(np.transpose(images, (1, 2, 0)))
            plt.title(f'Predicted: {classes[predicted[0].item()]}, Actual: {classes[actual.item()]}')
            images, actual, *predicted = data_iterable.__next__()
            images = images/2 + 0.5 #un-normalize

