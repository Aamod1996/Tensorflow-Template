import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


class TemplateGenerator(Sequence):

    def __init__(self, config):
        self.config = config
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        start_idx = idx*self.batch_size
        
        for i in range(self.batch_size):
            pass
            
        batch_x, batch_y = None, None
        return batch_x, batch_y
