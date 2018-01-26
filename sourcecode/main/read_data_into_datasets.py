# -*- coding: utf-8 -*-

import  xml.dom.minidom
import os
import  numpy as np
from ops import *

def read_data(input_file,output_file):
    # 解析pupil.xml文件，得到txt文件
    # 打开xml文档
    dom = xml.dom.minidom.parse(input_file)

    # 得到文档元素对象
    root = dom.documentElement

    name = root.getElementsByTagName('Name')
    x = root.getElementsByTagName('X')
    y = root.getElementsByTagName('Y')
    r = root.getElementsByTagName('R')

    output = open(output_file, 'a')
    path = input_file[:-len('pupil.xml')]
    for (name_i, x_i, y_i, r_i) in zip(name, x, y, r):
        output.write(path+name_i.firstChild.data)
        output.write(' ')
        output.write(x_i.firstChild.data)
        output.write(' ')
        output.write(y_i.firstChild.data)
        output.write(' ')
        output.write(r_i.firstChild.data)
        output.write('\n')
    output.close()

def find_file(file_path, lis,post='pupil.xml'):
    #寻找pupil.xml文件，返回绝对路径
    ls = os.listdir(file_path)
    for i in ls:
        son_path = os.path.join(file_path,i)
        if os.path.isdir(son_path):
            find_file(son_path,lis,post)
        else:
            if i == post:
                lis.append(os.path.join(file_path,i))

def main(f_path = r'/media/zhangsp/C098926B98926028/zsp/shuju/DeepoonE3'):
    #print(os.getcwd())
    #取当前目录

    #print(os.path.dirname(os.getcwd()))
    #取当前目录上一级目录

    #print(os.path.dirname(os.path.dirname(os.getcwd())))
    # 取当前目录上两级目录

    datasets_file_path = os.path.dirname(os.path.dirname(os.getcwd()))+'/datasets/train.data'

    lis = []
    find_file(f_path,lis)
    #找到所有的pupil.xml',解析到'data.txt'
    output_file = 'data.txt'
    for element in lis:
        print(element)
        read_data(element, output_file)

    #读取出'data.txt'每一行并打乱,输出到'data1.txt'
    txt = open(output_file, 'r')
    shuff_txt = []
    while 1:
        line = txt.readline()
        if not line:
            break
        shuff_txt.append(line)
    txt.close()
    np.random.shuffle(shuff_txt)
    output_file1 = 'data1.txt'
    txt1 = open(output_file1, 'w')
    for element in shuff_txt:
        txt1.write(element)
    txt1.close()

    #依据'data1.txt'将数据写入到文件夹datasets
    file = open(output_file1)
    new_data_path = datasets_file_path
    opts = None  # tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path=new_data_path, options=opts)
    i = 1
    while 1:
        line = file.readline()
        if not line:
            break
        picture_path, picture_x, picture_y, picture_r = line.split()
        # print(picture_path)
        if float(picture_x) > 0:
            image_bytes, h, w, chanel = getImageData(img_path=picture_path, use_opencv=True)
            # print(h,w,chanel)
            features = genRecord(img_data=(image_bytes, h, w, chanel), label=(picture_x, picture_y, picture_r))

            example = tf.train.Example(features=features)
            # print(example.SerializeToString())
            writer.write(example.SerializeToString())  # 序列化为字符串
            # print("write picture :", i)
            # print(picture_path)
            i = i + 1
    writer.close()

    


if __name__ == '__main__':
    main()