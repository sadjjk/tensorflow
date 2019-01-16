
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#构造隐藏层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs



x_data = np.linspace(-2, 2, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.power(x_data,3) - np.square(x_data) + x_data - 0.5 + noise


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 1, 20, activation_function=tf.nn.tanh)
l2 = add_layer(l1, 20, 10, activation_function = tf.nn.tanh)
# add output layer
prediction = add_layer(l2, 10, 1, activation_function=None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# important step
sess = tf.Session()


init = tf.global_variables_initializer()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()



for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        # print(prediction_value)
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.5)

# plt.ioff()
# plt.show()






