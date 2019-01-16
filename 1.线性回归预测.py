import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

x_data = np.linspace(-5,5,100)

y_data = 2 * x_data + 0.5 + np.random.normal(0,1,x_data.shape)


# plt.show()

# 构建线性回归
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b 


#定义loss function
loss = tf.reduce_mean(tf.square(y_data - y))

#梯度下降寻优
train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#会话
sess = tf.Session() 
sess.run(tf.global_variables_initializer()) 


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x_data,y_data,'b*',label = 'Original data')
plt.title('Linear Regression')
plt.legend()
plt.ion()


for step in range(1000):
    sess.run(train)

    if step % 100 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        y_value = sess.run(y)
        # plot the prediction
        lines = ax.plot(x_data, y_value, 'r-', lw=2,label = 'Prediction line')
        plt.legend()
        plt.pause(0.5)

sess.close()




