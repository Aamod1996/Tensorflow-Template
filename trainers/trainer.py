from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class TemplateTrainer(BaseTrain):
    def __init__(self, model, data, optimizer, loss, generator, config):
        super().__init__(model, data, config)
        self.optimizer = optimizer
        self.loss = loss
        self.generator = generator

    def train_epoch(self):                    
        loss = 0

        for i, (batch_x, batch_y) in enumerate(self.generator):
            batch_loss = self.train_step(batch_x, batch_y)
            loss += batch_loss
        
        return loss / i 

    def train_step(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            y_pred = self.model(X_batch, training=True)
            loss = tf.reduce_mean(self.loss(batch_y, y_pred))
            
        gradients = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss