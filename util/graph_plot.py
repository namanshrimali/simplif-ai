import matplotlib.pyplot as plt

def plot_me(train_losses, train_acc, test_losses, test_acc):


    fig, axs = plt.subplots(2,2,figsize=(20,15), sharex= False, sharey= False)

    axs[0,0].plot(train_losses)
    axs[0,0].set_title("Training Loss")

    axs[0,1].plot(test_losses)   
    axs[0, 1].set_title("Test Loss")
    
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Testing Accuracy")
