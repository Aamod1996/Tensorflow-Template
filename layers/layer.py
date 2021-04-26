import tensorflow as tf
from tensorflow.keras.layers import Layer


class TemplateLayer(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self):
        super().build() # must be at the end
        
    def call(self):
        pass
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}