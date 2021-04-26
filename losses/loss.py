import tensorflow as tf
from tensorflow import keras 


class TemplateLoss(keras.losses.Loss):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        pass
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}