# -*- coding: utf-8 -*-
from model.model_ops import *
import os

keep_prob = tf.placeholder("float")
learn_rate = tf.placeholder("float")
picture = tf.placeholder("float", shape=[None,400,400,3])
class MyModel():
    def __init__(self,looP=20000000,input_Channel=3,layer1_Node_num = 16,layer2_Node_num = 32,
                 layer3_Node_num = 64,fulllayer1_Node_num = 512,fulllayer2_Node_num=512,batch_Size =10,
                 model_Save_path ='mynet/save_net.ckpt'):
        self.input_channel = input_Channel
        self.layer1_node_num = layer1_Node_num
        self.layer2_node_num = layer2_Node_num
        self.layer3_node_num = layer3_Node_num
        self.fulllayer1_node_num = fulllayer1_Node_num
        self.fulllayer2_node_num = fulllayer2_Node_num
        self.model_save_path = model_Save_path
        self.batch_size  = batch_Size
        self.loop = looP


    def genModel(self,x, y_):
        # 第一层卷积(3 #400x400->8 #400x400)
        # 首先在每个5x5网格中，提取出24张特征图。其中weight_variable中前两维是指网格的大小，第三维的1是指输入通道数目，
        # 第四维的32是指输出通道数目（也可以理解为使用的卷积核个数、得到的特征图张数）。每个输出通道都有一个偏置项，因此偏置项个数为32。
        w_conv1 = weight_variable([3, 3, self.input_channel, self.layer1_node_num])
        b_conv1 = bias_variable([self.layer1_node_num])
        # 为了使之能用于计算，我们使用reshape将其转换为四维的tensor，其中第一维的－1是指我们可以先不指定，
        # 第二三维是指图像的大小，第四维对应颜色通道数目，灰度图对应1，rgb图对应3.
        #x_image = tf.reshape(x, [-1, 400, 400, 3])
        # 而后，我们利用ReLU激活函数，对其进行第一次卷积。
        h_conv1 = tf.nn.relu(conv2d(x, w_conv1,[1,1,1,1]) + b_conv1)

        # 第一次池化(8 #400x400->8 #200x200)
        # 比较容易理解，使用2x2的网格以max pooling的方法池化。
        h_pool1 = max_pool_2x2(h_conv1)
        #h_pool1 = h_conv1

        # 第二层卷积与第二次池化(8 #200x200->16 #200x200->16 #100x100)
        # 与第一层卷积、第一次池化类似的过程。

        w_conv2 = weight_variable([3, 3, self.layer1_node_num, self.layer2_node_num])
        b_conv2 = bias_variable([self.layer2_node_num])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2,[1,1,1,1]) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        #print(np.shape(h_pool2))

        # 第三层卷积与第三次池化(16 #100x100->32 #100x100->32 #50x50)
        # 与第一层卷积、第一次池化类似的过程。

        w_conv3 = weight_variable([2, 2, self.layer2_node_num, self.layer3_node_num])
        b_conv3 = bias_variable([self.layer3_node_num])

        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3,[1,1,1,1]) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        #print(np.shape(h_pool2))

        # 密集连接层
        # 此时，图片是200x200的大小。我们在这里加入一个有512个神经元的全连接层。
        # 之后把刚才池化后输出的张量reshape成一个一维向量，再将其与权重相乘，加上偏置项，再通过一个ReLU激活函数。
        w_fc1 = weight_variable([50 * 50 * self.layer3_node_num, self.fulllayer1_node_num])
        b_fc1 = bias_variable([self.fulllayer1_node_num])

        h_pool3_flat = tf.reshape(h_pool3, [-1, 50 * 50 * self.layer3_node_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

        w_fc2 = weight_variable([self.fulllayer1_node_num, self.fulllayer1_node_num])
        b_fc2 = bias_variable([self.fulllayer2_node_num])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

        # Dropout 这是一个比较新的也非常好用的防止过拟合的方法
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        # 回归与输出 应用了简单的softmax，输出。
        w_fc3 = weight_variable([self.fulllayer2_node_num, 3])
        b_fc3 = bias_variable([3])

        #y_conv = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        y_conv = tf.matmul(h_fc2_drop, w_fc3) + b_fc3

        # 计算均方误差的代价函数
        #print(np.shape(y_),np.shape(y_conv))
        Mse = tf.reduce_mean(tf.square(y_ - y_conv))
        # 使用优化算法使得代价函数最小化
        train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(Mse)
        #Mse = tf.reduce_mean(tf.cast(Mse, "float"))
        return Mse,train_step,y_conv,y_,b_fc3

    def train(self,data_path):
        int_h, int_w, int_chanel, image, float_x, float_y, float_r = read_and_decode(data_path)

        # 可以随机打乱输入
        _, _, _, image, y = tf.train.shuffle_batch([int_h, int_w, int_chanel, image, [float_x, float_y, float_r]],
                                                   batch_size=self.batch_size, capacity=400,
                                                   min_after_dequeue=5)

        #late = MyModel()
        model, train_step, y_conv, y_1 ,b_fc3= self.genModel(image, y)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # 线程需要同步
            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord)
            # threads = tf.train.start_queue_runners(sess=sess)
            new_learn_rate = 1e-5
            rate = 10**(-4.0/(self.loop/5000))
            for i in range(self.loop):
                if (i+1)%2000 == 0:
                    new_learn_rate = new_learn_rate * rate
                    save_path = saver.save(sess, self.model_save_path)
                    # 输出保存路径
                    print('Save to path: ', save_path)
                # with tf.device("/gpu:0"):
                # mse,_=sess.run([model,train_step],feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})
                if i % 10 == 0:
                    mse, _ = sess.run([model, train_step], feed_dict={keep_prob: 1.0, learn_rate: new_learn_rate})
                    #print(sess.run(y_1))
                    #print(sess.run(y_conv, feed_dict={keep_prob: 1}))
                    print("step %d, training accuracy %g" % (i, mse))
                    print("###########")
                else:
                    mse, _ = sess.run([model, train_step], feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})

            coord.request_stop()
            coord.join(threads)

    # 恢复神经网络参数,进行预测
    def restore_and_predict(self,img):
        # 定义权重参数
        w_conv1 = weight_variable([3, 3, self.input_channel, self.layer1_node_num])
        b_conv1 = bias_variable([self.layer1_node_num])

        w_conv2 = weight_variable([3, 3, self.layer1_node_num, self.layer2_node_num])
        b_conv2 = bias_variable([self.layer2_node_num])

        w_conv3 = weight_variable([2, 2, self.layer2_node_num, self.layer3_node_num])
        b_conv3 = bias_variable([self.layer3_node_num])

        w_fc1 = weight_variable([50 * 50 * self.layer3_node_num, self.fulllayer1_node_num])
        b_fc1 = bias_variable([self.fulllayer1_node_num])

        w_fc2 = weight_variable([self.fulllayer1_node_num, self.fulllayer2_node_num])
        b_fc2 = bias_variable([self.fulllayer2_node_num])

        w_fc3 = weight_variable([self.fulllayer2_node_num, 3])
        b_fc3 = bias_variable([3])

        h_conv1 = tf.nn.relu(conv2d(img, w_conv1, [1, 1, 1, 1]) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, [1, 1, 1, 1]) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, [1, 1, 1, 1]) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        h_pool3_flat = tf.reshape(h_pool3, [-1, 50 * 50 * self.layer3_node_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

        y_conv = tf.matmul(h_fc2, w_fc3) + b_fc3

        with tf.Session() as sess:
            # 加载文件中的参数数据，会根据name加载数据并保存到变量W和b中
            saver = tf.train.Saver()
            save_path = saver.restore(sess, self.model_save_path)

            print(sess.run(b_fc2))
            print('predictor is :')
            print(sess.run(y_conv))




