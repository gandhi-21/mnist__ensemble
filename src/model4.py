import tensorflow as tf

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

        self.weight = tf.Variable(tf.random_normal([self.concatenated_layer.get_shape()[-1], 10]))
        self.bias = tf.Variable(tf.random_normal([self.label_size]))


    def build(self):
        
       
        out = tf.add(tf.matmul(self.concatenated_layer, self.weight), self.bias)
        
        predictions = tf.nn.softmax(out)

        return out, predictions