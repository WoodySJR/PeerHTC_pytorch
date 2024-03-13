from torch.utils.data import Dataset
import torch
import warnings
warnings.filterwarnings("ignore")

class MyDataset(Dataset):
    def __init__(self, path_x, path_y, device):
        self._X = torch.tensor(torch.load(path_x)).to(device)
        self._y = torch.tensor(torch.load(path_y)).to(device)
    def __getitem__(self, index):
        return self._X[index], self._y[index]
    def __len__(self):
        return len(self._X)