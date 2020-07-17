
def load_log():
    
    with open('run.log', 'r') as f:
        lines = f.readlines()
    
    di = {}
    for l in lines:
        l = l.split(' - ')[-1].replace('\n', '')
        k, v = l.split('_')

        if k == 'epoch':
            epoch = v
            di[epoch] = {'h': [], 'w': [], 'loss': []}
            continue

        di[epoch][k].append(float(v))
        
    return di
