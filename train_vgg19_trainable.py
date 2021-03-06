"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import utils
from PIL import Image
import cv2
import os
import numpy as np
import csv
'''
def get_image(dir_path):
    dirs = os.listdir(dir_path)
    son_dir_path = os.path.join(dir_path+'/'+dirs[0])
    files = os.listdir(son_dir_path)
    #Creation of the list
    name = os.path.join(son_dir_path,files[0])
    with Image.open(name) as raw_im:
      im=raw_im.resize((224,224))
      im.save(name,'JPEG')
    new_image=cv2.imread(name)

    if dirs[0]=='background':
        batch=new_image
        batch.resize(1,224,224,3)
        label=np.array([[1,0,0,0,0]])
        label.resize(1,5)
    elif dirs[0]=='car':
        batch=new_image
        batch.resize(1,224,224,3)
        label=np.array([[0,1,0,0,0]])
        label.resize(1,5)
    elif dirs[0]=='bus':
        batch=new_image
        batch.resize(1,224,224,3)
        label=np.array([[0,0,1,0,0]])
        label.resize(1,5)
    elif dirs[0]=='van':
        batch=new_image
        batch.resize(1,224,224,3)
        label=np.array([[0,0,0,1,0]])
        label.resize(1,5)
    else:
        batch=new_image
        batch.resize(1,224,224,3)
        label=np.array([[0,0,0,0,1]])
        label.resize(1,5)
    #batches = batch
    #labels = label
    counter=1
    for dir in dirs:
        for file in files[1:]:
            name=os.path.join(son_dir_path,file)
            raw_im = Image.open(name)
            im=raw_im.resize((224,224))
            im.save(name,'JPEG')
            new_image=cv2.imread(name)
            new_image.resize(1,224,224,3)
            if dir=='background':
                batch=np.append(batch,new_image,axis=0)
                label=np.append(label,[[1,0,0,0,0]],axis=0)
            elif dir=='car':
                batch=np.append(batch,new_image,axis=0)
                label=np.append(label,[[0,1,0,0,0]],axis=0)
            elif dir=='bus':
                batch=np.append(batch,new_image,axis=0)
                label=np.append(label,[[0,0,1,0,0]],axis=0)
            elif dir=='van':
                batch=np.append(batch,new_image,axis=0)
                label=np.append(label,[[0,0,0,1,0]],axis=0)
            else:
                batch=np.append(batch,new_image,axis=0)
                label=np.append(label,[[0,0,0,0,1]],axis=0)
            counter=counter+1
            #if counter==2000:
            #    batches=np.append(batches,batch,axis=0)
            #    labels=np.append(labels,label,axis=0)
            #    np.delete(batch,np.arange(1,2000),axis=0)
            #    np.delete(label,np.arange(1,2000),axis=0)
            #   print ('the number of images loaded : 2000')
            #   counter=1
            print (counter)
    return batch, label
'''

def get_image(batch_size,dir_path,images_list,labels):
    files = os.listdir(dir_path)
    _index = np.random.randint(0,150931,batch_size)
    #print(_index)
    image_batch=np.empty([1,224,224,3])
    label_batch=np.empty([1,5])

    for i in range(batch_size):
        new_image=cv2.imread(os.path.join(dir_path,files[_index[i]]))
        new_image=cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB)
        new_image=new_image/225
        new_image.resize(1,224,224,3)
        image_batch=np.append(image_batch,new_image,axis=0)
        new_name=files[_index[i]]
        new_name=new_name[:8]
        #print(new_name)
        num=images_list.index(new_name)
        #print (num)
        one_label=labels[num]
        one_label.resize(1,5)
        #print (one)
        label_batch=np.append(label_batch,one_label,axis=0)
    image_batch=np.delete(image_batch,0,0)
    label_batch=np.delete(label_batch,0,0)
    return image_batch,label_batch


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

print ('csv loading finish')

batch_size = 32
STEPS = 50000

#batch1,label1=get_image(batch_size, './train', images_list, labels)



with tf.device('/gpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, shape=(batch_size,224,224,3))
    true_out = tf.placeholder(tf.float32, shape=(batch_size,5))
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg16_1.npy')
    train_mode = tf.constant(True, dtype=tf.bool)
    vgg.build(images, train_mode)

    print('database prepared')

    sess.run(tf.global_variables_initializer())
    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)
    for i in range(STEPS):
        batch_train,label_train=get_image(batch_size, './train_resized_image', images_list, labels)
        if i%500 == 0:
            total_cost = sess.run(cost,feed_dict={images:batch_train,true_out:label_train})
            print("After %d training step(s),cost on training data is %g" % (i,total_cost))
        sess.run(train, feed_dict={images: batch_train, true_out: label_train, train_mode: True})
    
        
    # test classification again, should have a higher probability about tiger
    #prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    #utils.print_prob(prob[0], './synset.txt')

    # test save
    vgg.save_npy(sess, './vgg16_4.npy')
