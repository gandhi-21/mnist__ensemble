import tensorflow as tf

from helper import conv2d, max_pool_2x2, dropout_layer

class Model2():

    def __init__(self, X2, Y2, keep_prob2):
        self.image_size = 28
        self.label_size = 10

        self.learning_rate = 1e-4
        
        self.X2 = X2
        self.Y2 = Y2
        self.keep_prob2 = keep_prob2

        self.weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 48])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 48, 64])),
            'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
            'wd1': tf.Variable(tf.random_normal([2048, 256])),
            'out': tf.Variable(tf.random_normal([256, self.label_size]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([48])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bc3': tf.Variable(tf.random_normal([128])),
            'bd1': tf.Variable(tf.random_normal([256])),
            'out': tf.Variable(tf.random_normal([self.label_size]))
        }


    def build(self):
        self.X2 = tf.reshape(self.X2, shape=[-1, 28, 28, 1])

        conv1 = conv2d(self.X2, self.weights['wc1'], self.biases['bc1'])
        conv1 = max_pool_2x2(conv1)

        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        conv2 = max_pool_2x2(conv2)

        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'])
        conv3 = max_pool_2x2(conv3)

        conv3 = tf.layers.flatten(conv3)

        fc1 = tf.add(tf.matmul(conv3, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        predictions = tf.nn.softmax(out)

        return out, predictions
