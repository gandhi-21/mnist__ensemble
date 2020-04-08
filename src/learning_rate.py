import tensorflow as tf
import numpy as np
import pandas as pd

train_data = pd.read_csv("./input/mnist_train.csv")

y_train = train_data.iloc[:,0]
y_train = train_data.iloc[:,0]
x_train = train_data.drop(train_data.columns[0], axis=1)

# Pre process and augment the data
x_train = normalize_data(x_train)
y_train = one_hot_encode(y_train)

x_train, y_train = shuffle_data(x_train, y_train)

# Split the data into test and train
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

X = tf.placeholder(tf.float32, shape=[None, 28 * 28])
Y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

model1 = Model1(X, Y, keep_prob)
logits1, predictions1 = model1.build()
loss_op1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=model1.Y))
train_op1 = tf.train.AdamOptimizer(learning_rate=model1.learning_rate).minimize(loss_op1)
accuracy1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions1, 1), tf.argmax(model1.Y, 1)), tf.float32))


init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

rates = list()
t_loss = list()
t_acc = list()

epochs = 10

with tf.Session(config=config) as sess:

    sess.run(init)

    data_loader = data_iterator(train_x, train_y, test_x, test_y, 56)

    for i in range(epochs):

        learning_rate *= 1.1
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        while data_loader.current_index < len(train_x):
            batch_x, batch_y = data_loader.next_batch()

            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            loss = sess.run(loss, feed_dict={X: batch_x, Y: batch_y})
            acc = sess.run(acc, feed_dict={X: batch_x, Y: batch_y})
            if np.isnan(loss):
                loss = np.nan_to_num(loss)
        rates.append(learning_rate)
        t_loss.append(loss)
        t_acc.append(acc)
        print(f'epoch {i + 1}: learning rate = {learning_rate:.10f}, loss = {loss:.10f}')
    
    # Calculate the learning rate based on the biggest derivative betweeen the loss and learning rate
dydx = list(np.divide(np.diff(t_loss), np.diff(rates)))
start = rates[dydx.index(max(dydx))]
print("Chosen start learning rate:", start)
print()