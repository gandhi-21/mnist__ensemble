# Function to do early stopping

import tensorflow as tf
import pandas as pd
import numpy as np

from pre_process import normalize_data, shuffle_data, one_hot_encode
from data_iterator import data_iterator
from model1 import Model1
from model2 import Model2
from model3 import Model3
from model4 import Model4

from sklearn.model_selection import train_test_split

# Load the dataset
train_data = pd.read_csv("./input/mnist_train.csv")
test_data = pd.read_csv("./input/mnist_test.csv")

y_train = train_data.iloc[:,0]
x_train = train_data.drop(train_data.columns[0], axis=1)

y_test = test_data.iloc[:,0]
x_test = test_data.drop(test_data.columns[0], axis=1)

# Pre process and augment the data
x_train = normalize_data(x_train)
y_train = one_hot_encode(y_train)

x_test = normalize_data(x_test)
y_test = one_hot_encode(y_test)

x_train, y_train = shuffle_data(x_train, y_train)
x_test, y_test = shuffle_data(x_test, y_test)

# Split the data into test and train
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

# Get a data iterator
# data_loader = data_iterator(train_x, train_y, test_x, test_y, 128)

X = tf.placeholder(tf.float32, shape=[None, 28 * 28])
Y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

X2 = tf.placeholder(tf.float32, shape=[None, 28 * 28])
Y2 = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob2 = tf.placeholder(tf.float32)

X3 = tf.placeholder(tf.float32, shape=[None, 28 * 28])
Y3 = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob3 = tf.placeholder(tf.float32)

Y4 = tf.placeholder(tf.float32, shape=[None, 10])
logitse1 = tf.placeholder(tf.float32, shape=[None, 10])
logitse2 = tf.placeholder(tf.float32, shape=[None, 10])

# Make model 1
model1 = Model1(X, Y, keep_prob)
logits1, predictions1 = model1.build()
loss_op1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=model1.Y))
train_op1 = tf.train.AdamOptimizer(learning_rate=model1.learning_rate).minimize(loss_op1)
accuracy1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions1, 1), tf.argmax(model1.Y, 1)), tf.float32))


# Make model 2
model2 = Model2(X2, Y2, keep_prob2)
logits2, predictions2 = model2.build()
loss_op2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=model2.Y2))
train_op2 = tf.train.AdamOptimizer(learning_rate=model2.learning_rate).minimize(loss_op2)
accuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions2, 1), tf.argmax(model2.Y2, 1)), tf.float32))


# Make model 3
model3 = Model3(X3, Y3, keep_prob3)
logits3, predictions3 = model3.build()
loss_op3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3, labels=model3.Y3))
train_op3 = tf.train.AdamOptimizer(learning_rate=model3.learning_rate).minimize(loss_op3)
accuracy3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions3, 1), tf.argmax(model3.Y3, 1)), tf.float32))

# # Make model 4
model4 = Model4(logitse1, logitse2, Y4)
logits4, predictions4 = model4.build()
loss_op4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits4, labels=model4.Y4))
train_op4 = tf.train.AdamOptimizer(learning_rate=model4.learning_rate).minimize(loss_op4)
accuracy4 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions4, 1), tf.argmax(model4.Y4, 1)), tf.float32))


# setup a session
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

epochs = 10

best, no_change = 0
limit = 4

# run the session
with tf.Session(config=config) as sess:

    sess.run(init)

    for epoch in range(epochs):

        print(f"training epoch {epoch}")

        data_loader = data_iterator(train_x, train_y, test_x, test_y, 128)

        while data_loader.current_index < len(train_x):
            # Run the training operation here
            batch_x, batch_y = data_loader.next_batch()

            sess.run(train_op1, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            sess.run(train_op2, feed_dict={X2: batch_x, Y2: batch_y, keep_prob2: 0.5})
            sess.run(train_op3, feed_dict={X3: batch_x, Y3: batch_y, keep_prob3: 0.5})
            sess.run(train_op4, feed_dict={Y4: batch_y, keep_prob3: 0.5})
        # Test on the test split

        # loss1 = sess.run(loss_op1, feed_dict={X: test_x, Y: test_y, keep_prob: 1.0})
        # loss2 = sess.run(loss_op2, feed_dict={X2: test_x, Y2: test_y, keep_prob2: 1.0})

        # print(f"Loss of network 1: {loss1}")
        # print(f"Loss of network 2: {loss2}")

        acc1 = sess.run(accuracy1, feed_dict={X: test_x, Y: test_y, keep_prob: 1.0})
        acc2 = sess.run(accuracy2, feed_dict={X2: test_x, Y2: test_y, keep_prob2: 1.0})
        acc3 = sess.run(accuracy3, feed_dict={X3: test_x, Y3: test_y ,keep_prob3: 1.0})
        acc4 = sess.run(accuracy4, feed_dict={logitse1: logits1, logitse2: logits2, Y4: batch_y})

        last_acc = [acc1, acc2, acc3, acc3]

        print(f"Accuracy of network 1: {acc1}")
        print(f"Accuracy of network 2: {acc2}")
        print(f"Accuracy of network 3: {acc3}")
        print(f"Accuracy of ensemble network: {acc4}")

        if best < max(last_acc):
            best = max(last_acc)
        else:
            no_change += 1

        if no_change >= limit:
            print(f"Early Stopping at epoch {epoch}")
            break

        print(f"trained epoch {epoch}")
        # Print accuracy and loss
    