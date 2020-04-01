import tensorflow as tf

from helper import conv2d, max_pool_2x2, dropout_layer

class Model1():
    
    def __init__(self, X, Y, keep_prob):
        self.image_size = 28
        self.label_size = 10

        self.learning_rate = 1e-4
        
        self.X = X
        self.Y = Y
        self.keep_prob = keep_prob

        self.weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            'out': tf.Variable(tf.random_normal([1024, self.label_size]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.label_size]))
        }
    

    def build(self):
        self.X = tf.reshape(self.X, shape=[-1, 28, 28, 1])

        conv1 = conv2d(self.X, self.weights["wc1"], self.biases['bc1'])
        conv1 = max_pool_2x2(conv1, ksize=2)

        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        conv2 = max_pool_2x2(conv2, ksize=2)

        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, self.keep_prob)
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        predictions = tf.nn.softmax(out)
        
        return out, predictions

    
