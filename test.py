# -*- coding: utf-8 -*-

import tensorflow as tf

import vgg19_trainable as vgg19
import utils

img1 = utils.load_image("./tiger.jpeg")
img1_true_result = [0,1,0,0,0]  # 1-hot result for tiger

batch1 = img1.reshape((1, 224, 224, 3))

with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 5])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19()
    vgg.build(images, train_mode)
    print('build finished')
    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # test classification
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    print (prob)
    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    print(prob)
    
    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    print(prob)

    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    print(prob)
    # test save
    vgg.save_npy(sess, './test-save.npy')