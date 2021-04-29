from tqdm import tqdm
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from base.base_train import BaseTrain


class Trainer(BaseTrain):
    def __init__(self, model, generator, config):
        super().__init__(model, generator, config)
        self.optimizer = Adam(learning_rate=self.config.learning_rate)
        self.loss = SparseCategoricalCrossentropy()
        self.generator = generator

    def train_epoch(self):                    
        loss = 0

        for i, (batch_x, batch_y) in enumerate(self.generator):
            batch_loss = self.train_step(batch_x, batch_y)
            loss += batch_loss
        
        return loss

    def train_step(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_x, training=True)
            loss = tf.reduce_mean(self.loss(batch_y, y_pred))
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    def val_epoch(self):
        loss = 0

        for i, (batch_x, batch_y) in enumerate(self.generator):
            batch_loss = self.val_step(batch_x, batch_y)
            loss += batch_loss
        
        return loss
    
    def val_step(self, batch_x, batch_y):
        y_pred = self.model(batch_x, training=True)
        loss = tf.reduce_mean(self.loss(batch_y, y_pred))
            
        return loss
        