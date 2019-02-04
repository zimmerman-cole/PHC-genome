import pandas as pd
import numpy as np
from .utils import one_hot_encode

class DataLoader(object):
    
    __targets__ = ['population', 'Geographic origin', 'Region', 'Pop7Groups', 'Sex']
    
    def __init__(self, target='Pop7Groups', make_bernoulli=True):
        self.data = pd.read_csv('./reduced_data.csv').set_index('CEPH ID')
        self.num_dims = 1000
        self.num_data = self.data.values.shape[0]
        self.make_bernoulli = make_bernoulli
        
        assert target in DataLoader.__targets__, "Target must be in %s." % DataLoader.__targets__
        self.target = target
        
        self.target_idx = dict()
        for t in self.data[target].unique():
            self.target_idx[t] = self.data.index[self.data[target] == t]
        self.num_targets = len(self.target_idx.keys())
        self.target_counters = {k: 0 for k in self.target_idx.keys()}
            
    def load_batches(self, batch_size, do_resampling=False, p=None, one_hot_encode_y=True):
        if do_resampling:
            if p is None:
                p = {k: (1. / self.num_targets) for k in self.target_idx.keys()}

            raise NotImplementedError
            
        else:
            ix = np.random.choice(self.num_data, replace=False, size=self.num_data)
            num_batches = int(np.ceil(self.num_data / batch_size))
            
            for batch_num in range(0, num_batches):
                b = batch_num * batch_size
                e = b + batch_size
                ix_batch = ix[b:e]
                
                X = self.data.values[ix_batch, :self.num_dims].astype(np.int8)
                if self.make_bernoulli:
                    X = X / 2.
                
                y = self.data[self.target].values[ix_batch]
                y, _ = one_hot_encode(y)
                
                yield (X, y)
                
                
                
                
                
                
                
                
                
pass