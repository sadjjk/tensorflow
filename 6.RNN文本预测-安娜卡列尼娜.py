import time
import numpy as np
import tensorflow as tf



#1.数据预处理


#1.1 导入小说并建立单个字母与数字映射
def encode_letter_int(text_path):

    with open(text_path, 'r') as f:
        text=f.read()

    letter = set(text)
    letter_to_int = {c: i for i, c in enumerate(letter)}
    int_to_letter = dict(enumerate(letter))
    encoded = np.array([letter_to_int[c] for c in text], dtype=np.int32)
    return letter_to_int,int_to_letter,encoded

#OR 导入小说并建立单词与数字映射
def encode_vocab_int(text_path):
    pass



#1.2 mini-batch
def get_batches(arr,n_seqs=20,n_steps=20):
    '''
    n_seqs: 一个batch中序列个数
    n_steps: 单个序列包含的字符数
    '''

    batch_size = n_seqs * n_steps
    arr = arr[:len(arr) // batch_size * batch_size ]

    arr = arr.reshape((n_seqs, -1))

    #举例 input：我喜欢你 targets：喜欢你我
    for n in range(0, arr.shape[1], n_steps):
        # inputs
        x = arr[:, n:n+n_steps]
        # targets
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

#2 tensorflow搭建
#2.1 输入层

def build_inputs(num_seqs,num_steps):
    '''
    num_seqs: 每个batch中的序列个数
    num_steps: 每个序列包含的字符数
    '''

    inputs = tf.placeholder(tf.int32,shape=(num_seqs,num_steps),name='inputs')
    targets = tf.placeholder(tf.int32,shape=(num_seqs,num_steps),name='targets')

    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    return inputs, targets, keep_prob

#2.2 lstm层
def build_lstm(lstm_size,num_layers,batch_seq_size,keep_prob):
    '''
    keep_prob
    lstm_size: lstm隐层中结点数目
    num_layers: lstm的隐层数目
    batch_seq_size: 每个batch中的序列个数
    '''

    def drop():
        #构建一个基本lstm单元
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

        #添加dropout
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm,keep_prob)

        return drop     

    #堆叠
    cell = tf.nn.rnn_cell.MultiRNNCell([drop() for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_seq_size,tf.float32)

    return cell,initial_state

#2.3 输出层
def build_output(lstm_output,lstm_size,out_size):
    '''
    lstm_output: lstm层的输出结果
    lstm_size: lstm隐层中结点数目
    out_size: softmax层的size    
    '''

    x = tf.reshape(lstm_output,[-1,lstm_size])

    # with tf.variable_scope('softmax'):
    softmax_w = tf.Variable(tf.truncated_normal([lstm_size,out_size],stddev = 0.1))
    softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x,softmax_w) + softmax_b

    out = tf.nn.softmax(logits,name='predictions')

    return out,logits

#2.4 loss函数
def build_loss(logits,targets,lstm_size,num_classes):

    y_one_hot = tf.one_hot(targets,num_classes)
    y_reshaped = tf.reshape(y_one_hot,logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits= logits,labels =y_reshaped)
    loss = tf.reduce_mean(loss)

    return loss 

#2.5 优化器Optimizer + 梯度裁剪gradient clippling
def build_optimizer(loss,learning_rate,grad_clip):

    tvars = tf.trainable_variables()
    grads,_ = tf.clip_by_global_norm(tf.gradients(loss,tvars),grad_clip)
    trian_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = trian_op.apply_gradients(zip(grads,tvars))

    return  optimizer



#3 模型组合
class RNN:
    def __init__(self,num_classes,
                    batch_seq_size = 20,
                    num_steps = 20,
                    lstm_size = 128,
                    num_layers = 2,
                    learning_rate = 0.001,
                    grad_clip = 5,
                    sampling = False):

        if sampling == True:
            batch_seq_size,num_steps = 1,1  #测试样本使用

        #输入层
        self.inputs,self.targets,self.keep_prob = build_inputs(batch_seq_size,num_steps)

        #LSMT层
        cell,self.initial_state = build_lstm(lstm_size,
                                             num_layers,
                                             batch_seq_size,
                                             self.keep_prob)

        x_one_hot = tf.one_hot(self.inputs, num_classes)

        outputs,state = tf.nn.dynamic_rnn(cell,x_one_hot,initial_state=self.initial_state)
        self.final_state = state

        #预测
        self.predictions,self.logits = build_output(outputs,lstm_size,num_classes)

        #loss & optimizer
        self.loss = build_loss(self.logits,self.targets,lstm_size,num_classes)
        self.optimizer = build_optimizer(self.loss,learning_rate,grad_clip)


batch_seq_size = 20
num_steps = 30

keep_prob = 0.5

letter_to_int,int_to_letter,encoded = encode_letter_int('anna.txt')

epochs = 20
# 每n轮进行一次变量保存
save_every_n = 200

model = RNN(num_classes=len(letter_to_int),batch_seq_size=batch_seq_size,num_steps=num_steps)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    counter = 0
    # for e in range(epochs):
        # Train network
    new_state = sess.run(model.initial_state)
    loss = 0
    for x, y in get_batches(encoded, batch_seq_size, num_steps):
        counter += 1
        start = time.time()
        feed = {model.inputs: x,
                model.targets: y,
                model.keep_prob: keep_prob,
                model.initial_state: new_state}

        # print(feed)
        # break
        # sess.run(model.final_state,feed_dict=feed)
        # sess.run(model.optimizer,feed_dict=feed)
        # batch_loss = sess.run(model.loss,feed_dict=feed)
        batch_loss, _, _ = sess.run([model.loss,
                                             model.final_state,
                                             model.optimizer],
                                             feed_dict=feed)

        end = time.time()
        # control the print lines
        if counter % 10 == 0:
            print(
                  # '轮数: {}/{}... '.format(e+1, epochs),
                  '训练步数: {}... '.format(counter),
                  '训练误差: {:.4f}... '.format(batch_loss),
                  '{:.4f} sec/batch'.format((end-start)))
    #
    #     if (counter % save_every_n == 0):
    #             saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    #
    # saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    #
    #
    #