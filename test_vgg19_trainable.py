"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import utils
import numpy as np
import os
import csv

with open('train_label.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    images_list=[]
    labels=np.empty([1,5])

    for row in readCSV:
        image = row[0]
        label_raw = row[1]
        #print (label_raw)
        if label_raw=='background':
            labels=np.append(labels,[[1,0,0,0,0]],axis=0)
            #label.resize(1,5)
        elif label_raw=='car':
            labels=np.append(labels,[[0,1,0,0,0]],axis=0)
            #label.resize(1,5)
        elif label_raw=='bus':
            labels=np.append(labels,[[0,0,1,0,0]],axis=0)
            #label.resize(1,5)
        elif label_raw=='work_van':
            labels=np.append(labels,[[0,0,0,1,0]],axis=0)
            #label.resize(1,5)
        else:
            labels=np.append(labels,[[0,0,0,0,1]],axis=0)
            #label.resize(1,5)

        images_list.append(image)
labels=np.delete(labels,0,0)
print (labels.shape)


with tf.device('/gpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 1000])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg16_4.npy')
    vgg.build(images, train_mode)

    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())
    correct=0
    count=0
    
    
    #load image
    dir_path='./train_resized_image'
    files = os.listdir(dir_path)
    
    for each in files:
        path=os.path.join(dir_path,each)
        img1 = utils.load_image(path)
        batch1 = img1.reshape((1, 224, 224, 3))
        each=each[:8]
        num=images_list.index(each)
        one_label=labels[num]
        # test classification
        prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        index=np.where(np.max(prob[0])==prob[0])
        if one_label[index]==1:
            correct=correct+1
        count=count+1
    print (correct)
    print (count)
