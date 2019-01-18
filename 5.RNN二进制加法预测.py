import tensorflow as tf
import numpy as np


#===============1.数据准备===============
#把十进制数字转成二进制,不足指定长度在开头补零
def num_to_binary(num,binary_length):
    binary_num = np.array([int(i) for i in (bin(num)[2:])])
    binary_num = np.hstack(((np.zeros(binary_length - len(binary_num))),binary_num))
    return binary_num


def binary_generation(numbers,binary_length):
    binary_x = np.array([num_to_binary(num,binary_length) for num in numbers])
    
    #左右翻转 为了输入从低位开始训练
    #如指定二进制长度为4   列表[1,4,8]进行如下转换
    #    [[0,0,0,1],               [[1,0,0,0],
    #    [0,1,0,0],    =======>    [0,0,1,0],
    #    [1,0,0,0]]                [0,0,0,1]]
    binary_x = np.fliplr(binary_x)
    return binary_x


def binary_to_num(binary_array):
    # 将一个二进制数组转为整数
    
    out = 0
    for index, x in enumerate(reversed(binary_array)):
        out += x*pow(2, index)
    return int(out)

def batch_generation(batch_size, binary_length):
    
    # 随机生成batch_size个数
    n1 = np.random.randint(0, pow(2,binary_length)//2, batch_size)
    n2 = np.random.randint(0, pow(2,binary_length)//2, batch_size)
    
    # 计算加法结果
    add = n1 + n2

    # int to binary
    binary_n1 = binary_generation(n1, binary_length)
    binary_n2 = binary_generation(n2, binary_length)
    batch_add = binary_generation(add, binary_length)

    # 堆叠，因为网络的输入是2个二进制
    batch_stack = np.dstack((binary_n1, binary_n2))

    return batch_stack,batch_add, n1, n2, add



#====================2.TensorFlow===============    

batch_size = 100  #批次数量

binary_length = 10 #二进制长度

lstm_size = 20 #隐藏层神经元数量

lstm_layers_num = 2 #隐藏层层数

#输入
input_x = tf.placeholder(tf.float32,[None,binary_length,2],name = 'input_x') # 有两个输入 即a+b

#输出
output_y = tf.placeholder(tf.float32,[None,binary_length],name='output_y')

#dropout参数
keep_prob = tf.placeholder(tf.float32,name='keep_prob')

#创建模型 

def lstm_cell():
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    #dropout 
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob = keep_prob)
    return drop

cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(lstm_layers_num)])

# # 初始状态，可以理解为初始记忆
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, final_state = tf.nn.dynamic_rnn(cell, input_x, initial_state=initial_state)


#输出层
weights = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.01))
bias = tf.zeros([1])

# [batch_size, lstm_size*binary_dim] ==> [batch_size*binary_dim, lstm_size]
outputs = tf.reshape(outputs, [-1, lstm_size])

# 得到输出, logits大小为[batch_size*binary_dim, 1]

logits = tf.sigmoid(tf.add(tf.matmul(outputs, weights),bias))

# [batch_size*binary_dim, 1] ==> [batch_size, binary_dim]
predictions = tf.reshape(logits, [-1, binary_length])

#损失函数
cost = tf.losses.mean_squared_error(output_y, predictions)
optimizer = tf.train.AdamOptimizer().minimize(cost)
accuracy = tf.metrics.accuracy(output_y,tf.round(predictions))[1]



# 训练
steps = 20000
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    for i in range(steps):
        # 获取训练数据
        train_x, train_y,_,_,_ = batch_generation(batch_size, binary_length)
        _, loss,train_accuracy = sess.run([optimizer, cost, accuracy], feed_dict={input_x:train_x,output_y:train_y, keep_prob:0.5})

        if i % 1000  == 0:
            print('Iter:{}, Loss:{},Accuracy:{:.2f}'.format(i, loss,train_accuracy)) 
            
            
#   训练结束，进行测试
    batch_stack,batch_add, n1, n2, add = batch_generation(batch_size, binary_length)
    result,test_accuracy = sess.run([predictions,accuracy], feed_dict={input_x:batch_stack, output_y:batch_add, keep_prob:1.0})

    print('Test Accuracy:{}'.format(test_accuracy)) 
#   左右翻转二进制数组。因为输出的结果是低位在前，而正常的表达是高位在前，因此进行翻转
    result = np.fliplr(np.round(result))

    for  pre_x, pre_y, a, b, add_num in zip(np.fliplr(batch_stack), result, n1, n2, add):
        print('{}:{}'.format(pre_x[:,0], a))
        print('{}:{}'.format(pre_x[:,1], b))
        print('{}: 预测值{} 实际值{}\n'.format(pre_y, binary_to_num(pre_y),add_num))
        
        


