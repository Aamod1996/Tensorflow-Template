from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization
import tensorflow as tf
from functools import partial

#Define regularized layers
RegularizedDense = partial(Dense,
                           kernel_initializer="he_uniform")

RegularizedConv2D = partial(Conv2D,
                            strides=(1,1),
                            kernel_initializer="he_uniform")


class HappyHouse(tf.keras.Model):
    
    def __init__(self,input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        #Define the layers
        self.conv1 = RegularizedConv2D(32, (7, 7), name='conv0', input_shape=self.input_dim)
        self.bn1 = BatchNormalization(axis=3, name='bn0')
        self.pool1 = MaxPooling2D((2, 2), name='max_pool0')
        self.conv2 = RegularizedConv2D(64, (5, 5), name='conv1')
        self.bn2 = BatchNormalization(axis=3, name='bn1')
        self.pool2 = MaxPooling2D((2, 2), name='max_pool1')
        self.linear1 = ResidualBlock(2, [100, 100], name="rb1")
        self.out = Dense(self.output_dim, kernel_initializer="glorot_uniform", name="out")
        self.relu = Activation("relu")
        self.sigmoid = Activation("sigmoid")
        
    #Forward pass    
    def call(self, inputs):
        
        x = self.relu(self.pool1(self.bn1(self.conv1(inputs))))
        x = self.relu(self.pool2(self.bn2(self.conv2(x))))
        x = Flatten()(x)
        x = self.relu(self.linear1(x))
        out = self.sigmoid(self.out(x))
        
        return out
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "ResidualBlock": ResidualBlock}