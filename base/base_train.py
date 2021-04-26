import time
import tensorflow as tf


class BaseTrain:
    
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        
    def train(self):
        print("Starting training...\n")
        start_time = time.time()
        
        losses = []
        
        for epoch in range(self.config["num_epochs"]):
            loss = self.train_epoch()
            
            if epoch % self.config["print_every"] == 0:
                print("Loss: {}, time: {}".format(losses[epoch],
                                                  time.time()-start_time))
          
            losses.append(loss)
            
        return losses
          
    def train_epoch(self):
        raise NotImplementedError
    
    def train_step(self):
        raise NotImplementedError