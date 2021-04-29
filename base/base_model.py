import os
import tensorflow as tf
from tensorflow.keras import Model


class BaseModel(Model):
    
    def __init__(self, config):
        super(BaseModel, self).__init__(**kwargs)
        self.config = config
        
    def save_params(self):
        path = os.path.join("checkpoints", self.config.name, self.config.version)
        os.makedirs(path, exist_ok=True)
        self.save_weights(path)
        
    def load_params(self):
        path = os.path.join("checkpoints", self.config.name, self.config.version)
        self.load_weights(path)
        