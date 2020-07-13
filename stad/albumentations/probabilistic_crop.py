import numpy as np
from albumentations.core.transforms_interface import DualTransform

class ProbabilisticCrop(DualTransform):
    
    def __init__(self, 
                 height: int, 
                 width: int, 
                 anomaly_map: np.array,
                 always_apply: bool=False, 
                 p: float=1.0):
        
        super(ProbabilisticCrop, self).__init__(always_apply, p)
        self.half_h = height // 2
        self.half_w = width // 2
        self.anomaly_map = anomaly_map
        
        
    def apply(self, img, **params):

        a = np.arange(np.prod(self.anomaly_map.shape))
        p = self.anomaly_map.flatten()
        p = p / p.sum()
        sample = np.random.choice(a=a, size=1, p=p)[0]
        h, w = divmod(sample, self.anomaly_map.shape[1])
        h = max(h, self.half_h)
        h = min(h, img.shape[0] - self.half_h)
        w = max(w, self.half_w)
        w = min(w, img.shape[1] - self.half_w)
        
        return img[h-self.half_h:h+self.half_h, w-self.half_w:w+self.half_w, :]
