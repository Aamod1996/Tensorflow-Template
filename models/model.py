import tensorflow as tf
from tensorflow.keras import Model
from base.base_model import BaseModel
from preprocessors.preprocessor import Preprocessor


class TemplateModel(BaseModel, Model):
    
    def __init__(self, config, **kwargs):
        Model.__init__(**kwargs)
        BaseModel.__init__(config)
        self.preprocessor = Preprocessor()
        
    def call(self, inputs):
        x = self.preprocess(inputs)
        pass
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
    
    def preprocess(self, inputs):
        return self.preprocess.process(inputs)