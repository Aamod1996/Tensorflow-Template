import numpy as np
import os
import pickle
import tensorflow as tf
from base.base_predictor import BasePredictor
from models.model import CharacterModel


class Predictor(BasePredictor):
    
    def __init__(self, config, **kwargs):
        super(Predictor, self).__init__(config, **kwargs)
        self.load_model()
        
    def predict(self, inputs):
        x = self.preprocess(inputs)
        y_pred = self.model.predict_classes(x)
        return y_pred
    
    def preprocess(self, inputs):
        pass
