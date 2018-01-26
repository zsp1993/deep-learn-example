# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import os
from model.model_ops import *
from model.mymodel import *
from ops import *

datasets_file_path = os.path.dirname(os.path.dirname(os.getcwd()))+'/datasets/train.data'
predictor_file_path = os.path.dirname(os.path.dirname(os.getcwd()))+'/datasets/example1.bmp'

def main(process='train',train_path = datasets_file_path,predict_path = predictor_file_path):
    model1 = MyModel()

    if process=='train':
        model1.train(train_path)
    if process=='predictor':
        raw_bytes = cv2.imread(predict_path)
        #showBatchImage(raw_bytes)
        img = tf.reshape(raw_bytes, [1, 400, 400, 3]) / 255 * 2 - 1
        model1.restore_and_predict(img)

if __name__ == '__main__':
    #main('predictor')
    main('train')
