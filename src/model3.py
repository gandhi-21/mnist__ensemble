import tensorflow as tf

from helper import conv2d, max_pool_2x2, dropout_layer

class Model3():

    def __init__(self, X3, Y3, keep_prob3):
        self.image_size = 28
        self.label_size = 10

        self.learning_rate = 1e-4
        
        self.X3 = X3
        self.Y3 = Y3
        self.keep_prob3 = keep_prob3

        self.weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 32])),
            'wd1': tf.Variable(tf.random_normal([7*7*32, 256])),
            'out': tf.Variable(tf.random_normal([256, self.label_size]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([32])),
            'bd1': tf.Variable(tf.random_normal([256])),
            'out': tf.Variable(tf.random_normal([self.label_size]))
        }

    
    def build(self):
        self.X3 = tf.reshape(self.X3, shape=[-1, 28, 28, 1])

        conv1 = conv2d(self.X3, self.weights['wc1'], self.biases['bc1'])
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'])

        conv2 = max_pool_2x2(conv2)
        conv2 = dropout_layer(conv2, self.keep_prob3)

        conv2 = max_pool_2x2(conv2)
        conv2 = dropout_layer(conv2, self.keep_prob3)

        # Flatten
        conv3 = tf.layers.flatten(conv2)
        # Dense
        fc1 = tf.add(tf.matmul(conv3, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Dropout
        fc1 = dropout_layer(fc1, self.keep_prob3)
        # Dense
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        predictions = tf.nn.softmax(out)

        return out, predictions
