import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import models


class BasePredictor(Model):
    
    def __init__(self, config, **kwargs):
        super(BasePredictor, self).__init__(**kwargs)
        self.config = config 
        
    def load_model(self):
        print("Loading model...\n")
        path = os.path.join("saved_models", self.config.name, self.config.version)
        self.model = models.load_model(path)
        print("Model loaded...\n")
        