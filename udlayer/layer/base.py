"""Base classes for UDLayer."""


from abc import ABC, abstractmethod
import pickle


class BaseLayer(ABC):
    def __init__(self, name, year=None) -> None:
        super().__init__()
        self.name = name
        self.year = year
        self.data = None
    
    def save_layer(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load_layer(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_layer_data(self):
        return self.data
        
        
