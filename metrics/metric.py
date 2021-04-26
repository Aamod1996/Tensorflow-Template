import tensorflow as tf
from tensorflow import keras 


class TemplateMetric(keras.metrics.Metric):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        pass 
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}