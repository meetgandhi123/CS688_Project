import numpy as np
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len
    
    def __len__(self):
        return len(self.data)

class GoEmotionsSoft(Dataset):
    def __init__(self, X, Y, softlabels):
        self.data = X
        self.target = Y
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        self.soft_labels = softlabels.float()
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        soft_labels = self.soft_labels[index]
        return x, y, x_len, soft_labels, int(index)
    
    def __len__(self):
        return len(self.data)
