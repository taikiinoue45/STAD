from pathlib import Path


def clean_up():
    
    base = Path('.')
        
    [p.unlink() for p in base.glob('val/*_img.jpg')]
    [p.unlink() for p in base.glob('val/*_mask.png')]
    [p.unlink() for p in base.glob('val/*_anomaly_map.npy')]
    
    [p.unlink() for p in base.glob('test/*/*_img.jpg')]
    [p.unlink() for p in base.glob('test/*/*_mask.png')]
    [p.unlink() for p in base.glob('test/*/*_anomaly_map.npy')]
