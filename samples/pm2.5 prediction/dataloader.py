from torch.utils.data import Dataset, DataLoader

class CityDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = {'x': self.data[index], 'y': self.label[index]}
        return sample