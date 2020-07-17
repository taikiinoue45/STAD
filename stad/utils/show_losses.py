import matplotlib.pyplot as plt

from .load_log import load_log


def show_losses():
    
    log = load_log()
    
    epochs = []
    losses = []
    for epoch, di in log.items():
        epochs.append(int(epoch))
        losses.append(di['loss'])

    plt.plot(epochs, losses, color='k')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()    
    plt.savefig('loss.png')
    plt.close()
