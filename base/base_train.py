import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf


class BaseTrain:
    
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        
    def train(self):
        
        with tf.device(self.config.device):
            print("Starting training...\n")
            start_time = time.time()
            
            self.train_losses = []
            self.val_losses = []
            
            for epoch in range(self.config.num_epochs):
                train_loss = self.train_epoch()
                val_loss = self.val_epoch()
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                if epoch % self.config.print_every == 0:
                    print("Epoch: {}, Train Loss: {}, Val Loss: {}, Time: {}".format(
                        epoch+1, 
                        self.train_losses[epoch],
                        self.val_losses[epoch],
                        time.time()-start_time)
                        )

                    start_time = time.time()
                    
                if epoch % self.config.save_every == 0:
                    self.model.save_params()
                
            print("Training complete...\n")
            
            self.save_model()
            
            self.plot_losses()
                
    def plot_losses(self):
        plt.figure()
        plt.plot(range(self.config.num_epochs), self.train_losses, label="Training Loss")
        plt.plot(range(self.config.num_epochs), self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
    def save_model(self):
        print("Saving model...\n")
        path = os.path.join("checkpoints", self.config.name, self.config.version)
        os.makedirs(path, exist_ok=True)
        self.model.save(path)
          
    def train_epoch(self):
        raise NotImplementedError
    
    def train_step(self):
        raise NotImplementedError
    