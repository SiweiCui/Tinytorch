import numpy as np 
from tinytorch import cuda

class Dataset:
    def __init__(self, train = True, transform = None, target_transform = None) -> None:
        self.train = train 
        self.transform = transform 
        self.target_transform = target_transform 
        if self.transform is None:
            self.transform = lambda x: x 
        if self.target_transform is None:
            self.target_transform = lambda x: x 
        
        self.data = None 
        self.label = None 
        self.prepare() # 在prepare中载入数据

    def __getitem__(self, index):
        assert np.isscalar(index) # 只支持index是整数的情况
        if self.label is None:
            return self.transform(self.data[index]), None 
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index]) 
    
    def __len__(self):
        return len(self.data)
    
    def prepare(self):
        pass 

