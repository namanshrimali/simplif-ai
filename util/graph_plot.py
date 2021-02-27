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


def plot_graph(data: tuple, plot_title: str, axis_lables: tuple, save: bool):
    fig, ax = plt.subplots()
    ax.set(xlabel=axis_lables[0], ylabel=axis_lables[1], title=plot_title)
    ax.grid()
    ax.plot(data[0], data[1])
    plt.show() # dunno what it does, graphs plot without it too 
    if save:
        fig.savefig(f'{plot_title}.png')
        
    