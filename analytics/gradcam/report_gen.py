import matplotlib.pyplot as plt
from analytics.gradcam.gradcam_util import *
from util.un_normalize import UnNormalize
from util.get_batched_data import *

def generate_gradcam(model, test_loader, device, target_layers, mean, std, classes, total=25):

    testloader_iterable = iter(test_loader)
    data, target = next(testloader_iterable)[:total]
    
    data, target = data.to(device), target.to(device) # Sending to Gradcam

    gcam_layers, predicted_probs, predicted_classes = get_gradcam(data, target, model, device, target_layers)
    
    # get the denomarlization function
    unorm = UnNormalize(mean=mean, std=std)

    plt_gradcam(gcam_layers=gcam_layers, images=data, target_labels=target, predicted_labels= predicted_classes, class_labels= classes, un_normalize= unorm)

def get_misclassified_gradcam(model, misclassified, device, target_layers, mean, std, classes, total=25):
    
    data, target, _ = get_misclassified_as_batched_data(misclassified, batch_size = total)
    gcam_layers, _, predicted_classes = get_gradcam(data, target, model, device, target_layers)
    
    unorm = UnNormalize(mean=mean, std = std)
    
    plt_gradcam(gcam_layers=gcam_layers, images=data, target_labels=target, predicted_labels= predicted_classes, class_labels= classes, un_normalize= unorm)
