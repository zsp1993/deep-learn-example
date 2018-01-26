# -*- coding: utf-8 -*-

import sys, cv2
from PIL import Image
import tensorflow as tf
import  numpy as np


def getImageDataByCv(img_path):
    img = cv2.imread(img_path)
    # print (img)
    h, w, chanel = img.shape
    image_bytes = img.reshape([h * w * chanel]).tobytes()
    return image_bytes, h, w, chanel


def getImageDataByPIL(img_path):
    img = Image.open(img_path)
    """
    L (8-bit pixels, black and white)
    P (8-bit pixels, mapped to any other mode using a color palette)
    """
    image_bytes = img.convert("P").tobytes()
    return image_bytes, img.size[0], img.size[1]


def getImageData(img_path, use_opencv=True):
    if use_opencv:
        return getImageDataByCv(img_path)
    return getImageDataByPIL(img_path)


def genRecord(img_data, label):
    img_bytes = img_data[0]
    h = img_data[1]
    w = img_data[2]
    chanel = img_data[3]
    picture_x = float(label[0])
    picture_y = float(label[1])
    picture_r = float(label[2])

    """
    proto结构
    message Features {
        map<string, Feature> feature = 1;
    };
    message Feature {
      oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
      }
    };
    """
    features = tf.train.Features(feature={
        "h": tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
        "w": tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
        "chanel": tf.train.Feature(int64_list=tf.train.Int64List(value=[chanel])),
        'BytesList': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
        "picture_x": tf.train.Feature(float_list=tf.train.FloatList(value=[picture_x])),
        "picture_y": tf.train.Feature(float_list=tf.train.FloatList(value=[picture_y])),
        "picture_r": tf.train.Feature(float_list=tf.train.FloatList(value=[picture_r]))
    })
    return features


def showBatchImage(img_bytes):
    image = img_bytes
    cv2.imshow('test', image)
    cv2.waitKey(100)
    pass


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader(name=None, options=None)
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={'h': tf.FixedLenFeature([], tf.int64),
                                                 'w': tf.FixedLenFeature([], tf.int64),
                                                 'chanel': tf.FixedLenFeature([], tf.int64),
                                                 'BytesList': tf.FixedLenFeature([], tf.string),
                                                 "picture_x": tf.FixedLenFeature([], tf.float32),
                                                 "picture_y": tf.FixedLenFeature([], tf.float32),
                                                 "picture_r": tf.FixedLenFeature([], tf.float32),
                                                 }
                                       )

    raw_bytes = tf.decode_raw(features['BytesList'], tf.uint8)

    img = tf.reshape(raw_bytes, [400, 400, 3]) / 255 * 2 - 1
    x = tf.cast(features['picture_x'], tf.float32)
    y = tf.cast(features['picture_y'], tf.float32)
    r = tf.cast(features['picture_r'], tf.float32)

    return features['h'], features['w'], features['chanel'], img, x, y, r

