import tensorflow as tf
import numpy as np

from helper import conv2d, max_pool_2x2, dropout_layer

class Model4():

    def __init__(self, logits1, logits2, Y4):
        self.image_size = 28
        self.label_size = 10

        self.learning_rate = 1e-4

        self.logitse1 = logits1
        self.logitse2 = logits2

        self.Y4 = Y4

        self.concatenated_layer = tf.concat([self.logitse1, self.logitse2], axis=1)
        self.concatenated_layer = np.array([self.concatenated_layer, 0])

        print(f"shape of concatened layer {self.concatenated_layer.shape}")

        self.weight = tf.Variable(tf.random_normal([20, 20]))
        self.bias = tf.Variable(tf.random_normal([self.label_size]))


    def build(self):
        
       
        out = tf.add(tf.matmul(self.concatenated_layer, self.weight), self.bias)
        
        predictions = tf.nn.softmax(out)

        return out, predictions