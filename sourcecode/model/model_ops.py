# -*- coding: utf-8 -*-

import tensorflow as tf


# 初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    # 这是一个截断的产生正太分布的函数，
    # 就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
    return tf.Variable(initial)

# 初始化偏置项
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积过程
def conv2d(x,w,strides=[1,2,2,1]):
    return tf.nn.conv2d(x,w,
                        strides,padding='SAME')
#池化过程
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                        strides=[1,2,2,1],padding='SAME')

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader(name=None,options=None)
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={'h': tf.FixedLenFeature([], tf.int64),
                                           'w': tf.FixedLenFeature([], tf.int64),
                                            'chanel': tf.FixedLenFeature([],tf.int64),
                                            'BytesList' : tf.FixedLenFeature([], tf.string),
                                            "picture_x" : tf.FixedLenFeature([],tf.float32),
                                            "picture_y": tf.FixedLenFeature([], tf.float32),
                                            "picture_r": tf.FixedLenFeature([], tf.float32),
                                       }
                                       )

    raw_bytes = tf.decode_raw(features['BytesList'], tf.uint8)

    img = tf.reshape(raw_bytes, [400, 400, 3])/255*2-1
    x = tf.cast(features['picture_x'],tf.float32)
    y = tf.cast(features['picture_y'], tf.float32)
    r = tf.cast(features['picture_r'], tf.float32)
    #print("mark11111",img)

    return features['h'], features['w'],features['chanel'], img, x,y,r


