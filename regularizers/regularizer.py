import tensorflow as tf
from tensorflow import keras 


class TemplateRegularizer(keras.regularizers.Regularizer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self, weights):
        pass
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
    
