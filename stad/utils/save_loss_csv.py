import pandas as pd

from .load_log import load_log


def save_loss_csv():
    
    log = load_log()
    
    df = {'epoch': [], 'loss': []}
    for epoch, di in log.items():
        df['epoch'].append(int(epoch))
        df['loss'].append(di['loss'])

    df = pd.DataFrame(df)
    df.to_csv('loss.csv')