import matplotlib.pyplot as plt
from analytics.gradcam.gradcam_util import *
from util.un_normalize import UnNormalize

def generate_gradcam(model, data, device, target_layers, mean, std, classes):

    data, target = data.to(device), target.to(device) # Sending to Gradcam

    gcam_layers, predicted_probs, predicted_classes = get_gradcam(data, target, model, device, target_layers)
    
    # get the denomarlization function
    unorm = UnNormalize(mean=mean, std=std)

    plt_gradcam(gcam_layers=gcam_layers, images=data, target_labels=target, predicted_labels= predicted_classes, class_labels= classes, un_normalize= unorm)