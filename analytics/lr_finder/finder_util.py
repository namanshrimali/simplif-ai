from lr_finder import LRFinder

def find_me_ideal_lr(model, optimizer, criterion, trainloader, device, plot_graph = False, num_iter = 100, step_mode = "linear"):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(trainloader, val_loader=None, end_lr=1, num_iter=100, step_mode="linear")
    optimal_lr, _ = lr_finder.find_optimal_lr_loss()
    
    if plot_graph:
        lr_finder.plot(
            skip_start=10,
            skip_end=5,
            log_lr=True,
            show_lr=None,
            ax=None,
            suggest_lr=False,
        )
    lr_finder.reset()
    return optimal_lr
    