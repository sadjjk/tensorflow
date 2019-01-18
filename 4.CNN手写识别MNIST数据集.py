import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 


mnist = input_data.read_data_sets('mnist_data',one_hot = True)


#测试集数据
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]


# define placeholder for inputs to network


#增加的最后一维是颜色通道
#MNIST数据集灰色图片 所以是1 若为RGB三原色图片 则是3 
input_x = tf.placeholder(tf.float32,[None,28*28])

input_x_images = tf.reshape(input_x,[-1,28,28,1])

output_y = tf.placeholder(tf.int32,[None,10])



#构建卷积神经网络

#第1层卷积
conv1 = tf.layers.conv2d(
	inputs = input_x_images,  # [NUM,28,28,1]
	filters = 32,             # 卷积过滤器数量
	kernel_size = [5,5],      # 过滤器的维度(长宽)
	strides = 1,              # 步长
	padding = 'SAME' ,        # 补全填充
	activation = tf.nn.relu   # 激活函数
	)                         # 输出形状 [NUM,28,28,32]

#第1层池化(亚采样)
pool1 = tf.layers.max_pooling2d(
	inputs = conv1,           # 输入形状 [NUM,28,28,32]
	pool_size = [2,2],        # 过滤器大小
	strides = 2               # 步长 
	)                         # 输出形状 [NUM,14,14,32]

#第2层卷积
conv2 = tf.layers.conv2d(
	inputs = pool1,           # 输入形状 [NUM,14,14,32]
	filters = 64,             # 过滤器数量
	kernel_size = [5,5],      # 过滤器大小
	strides = 1 ,             # 步长
	padding = 'same',         # 补零填充
	activation = tf.nn.relu   # 激活函数
	)                         # 输出形状 [NUM,14,14,64]


#第2层池化
pool2 = tf.layers.max_pooling2d(
	inputs = conv2,           # 输入形状 [NUM,14,14,64]
	pool_size = [2,2],        # 过滤器大小
	strides = 2,              # 步长
	)                         # 输出形状 [NUM,7,7,64]


#扁平化 
flat = tf.reshape(pool2,[-1,7*7*64])  #输出形状 [NUM,3136]


#FC1 1024个神经元全连接
dense1 = tf.layers.dense(inputs = flat,
						units = 1024,
						activation = tf.nn.relu)
#dropout 
dropout = tf.layers.dropout(inputs=dense1,rate = 0.5)

#FC2  10个神经元全连接层
dense_pred = tf.layers.dense(inputs = dropout,
						 units = 10)

#计算误差 交叉熵 Cross entropy 

loss = tf.losses.softmax_cross_entropy(onehot_labels = output_y,logits = dense_pred)

#Adam 优化器
train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)


#精度 需要创建局部变量
accuracy = tf.metrics.accuracy(
			labels = tf.argmax(output_y,axis =1),
			predictions = tf.argmax(dense_pred,axis = 1))[1]


#创建会话
sess = tf.Session()

#初始化：全局和局部
init = tf.group(tf.global_variables_initializer(),
				tf.local_variables_initializer())

sess.run(init)


for i in range(1000):
	batch = mnist.train.next_batch(50)

	train_loss,train_op_value = sess.run([loss,train_op],
										{input_x:batch[0],output_y:batch[1]})


	if i % 100 == 0:
		test_accuracy = sess.run(accuracy,
								{input_x:test_x,output_y:test_y})
		print(("Sted=%d Train loss=%.4f Test accuracy=%.2f") % (i,train_loss,test_accuracy))

sess.close()