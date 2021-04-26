import tensorflow as tf
from tensorflow import keras 


class Regularizer(keras.regularizers.Regularizer):
    
    def __init__(self, factor, **kwargs):
        self.factor = factor
        super().__init__(**kwargs)
        
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "factor":self.factor}