import os
import tensorflow as tf


class BaseModel:
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def save_params(self, name="model", version="0001"):
        model_path = os.path.join(name, ".", version)
        print("Saving model weights...\n")
        self.save_weights(model_path, save_format="tf")
        print("Model weights saved...\n")
        
    def load_params(self, path="saved_models/model.0001"):
        print("Loading model weight...\n")
        self.load_weights(path)
        print("Model weights loaded...\n")

        