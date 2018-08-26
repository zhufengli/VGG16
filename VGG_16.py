from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

'''
some functions
'''


'''
Get images and gt information and reshape
'''
def get_image(dir_path):
    dirs = os.listdir(dir_path)
    for dir in dirs:
        son_dir_path = os.path.join(dir_path+'/'+dir)
        files = os.listdir(son_dir_path)
        #Creation of the list
        name = os.path.join(son_dir_path,files[0])
        raw_im = Image.open(name)
        im=raw_im.resize((224,224))
        im.save(name,'JPEG')
        new_image=cv2.imread(name)
        batch=new_image
        batch.resize(1,224,224,3)
        if dir=='background':
            label=np.array([[1,0,0,0,0]])
        elif dir=='car':
            label=np.array([[0,1,0,0,0]])
        elif dir=='bus':
            label=np.array([[0,0,1,0,0]])
        elif dir=='van':
            label=np.array([[0,0,0,1,0]])
        else:
            label=np.array([[0,0,0,0,1]])
        label.resize(1,5)
        
        for file in files[1:]:
            name=os.path.join(son_dir_path,file)
            raw_im = Image.open(name)
            im=raw_im.resize((224,224))
            im.save(name,'JPEG')
            new_image=cv2.imread(name)
            new_image.resize(1,224,224,3)
            batch=np.append(batch,new_image,axis=0)
            if dir=='background':
                label=np.array([[1,0,0,0,0]])
            elif dir=='car':
                label=np.array([[0,1,0,0,0]])
            elif dir=='bus':
                label=np.array([[0,0,1,0,0]])
            elif dir=='van':
                label=np.array([[0,0,0,1,0]])
            else:
                label=np.array([[0,0,0,0,1]])
            print (label.shape,batch.shape)
    return batch, label

'''
def VGG16 model
'''
def vgg_model(features, labels, mode):
    #input layer
	input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
	# Convolutional Layer #1
	conv1 = tf.nn.conv2d(
		inputs=input_layer,
		filters=(3, 3, 3, 64),
		strides=(1, 1, 1, 1),
		padding="VALID")
	relu1 = tf.nn.relu(conv1)
	# Convolutional Layer #2
	conv2 = tf.nn.conv2d(
		inputs=relu1,
		filters=(3, 3, 3, 64),
		strides=(1, 1, 1, 1),
		padding="SAME")
	relu2 = tf.nn.relu(conv2)
	# Pooling Layer #1
	pool1 = tf.nn.max_pool(inputs=relu2, pool_size=[1, 2, 2, 1], strides=2)
	# Convolutional Layer #3
	conv3 = tf.nn.conv2d(
		inputs=pool1,
		kernel_size=[3, 3, 64, 128],
		strides=(1, 1, 1, 1),
		padding="SAME")
	relu3 = tf.nn.relu(conv3)
	# Convolutional Layer #4
	conv4 = tf.nn.conv2d(
    	inputs=relu3,
    	kernel_size=[3, 3, 64, 128],
		strides=(1, 1, 1, 1),
    	padding="SAME")
	relu4 = tf.nn.relu(conv4)
	# Pooling Layer #2
	pool2 = tf.layers.max_pool(inputs=relu4, pool_size=[1, 2, 2, 1], strides=2)
	# Convolutional Layer #5
	conv5 = tf.nn.conv2d(
	   	inputs=pool2,
	   	kernel_size=[3, 3, 128, 256],
		strides=(1, 1, 1, 1),
	   	padding="SAME")
	relu5 = tf.nn.relu(conv5)
	# Convolutional Layer #6
	conv6 = tf.nn.conv2d(
	   	inputs=relu5,
	   	kernel_size=[3, 3, 128, 256],
		strides=(1, 1, 1, 1),
	   	padding="SAME")
	relu6 = tf.nn.relu(conv6)
	# Convolutional Layer #7
	conv7 = tf.nn.conv2d(
		inputs=relu6,
		kernel_size=[3, 3, 128, 256],
		strides=(1, 1, 1, 1),
		padding="SAME")
	relu7 = tf.nn.relu(conv7)
	# Pooling Layer #3
	pool3 = tf.layers.max_pool(inputs=relu7, pool_size=[1, 2, 2, 1], strides=2)
	# Convolutional Layer #8
	conv8 = tf.nn.conv2d(
	   	inputs=pool3,
	   	kernel_size=[3, 3, 256, 512],
		strides=(1, 1, 1, 1),
	   	padding="SAME")
	relu8 = tf.nn.relu(conv8)
	# Convolutional Layer #9
	conv9 = tf.nn.conv2d(
	   	inputs=relu8,
	   	kernel_size=[3, 3, 256, 512],
		strides=(1, 1, 1, 1),
	   	padding="SAME")
	relu9 = tf.nn.relu(conv9)
	# Convolutional Layer #10
	conv10 = tf.nn.conv2d(
	   	inputs=relu9,
	   	kernel_size=[3, 3, 256, 512],
		strides=(1, 1, 1, 1),
	   	padding="SAME")
	relu10 = tf.nn.relu(conv10)
	# Pooling Layer #4
	pool4 = tf.layers.max_pool(inputs=relu10, pool_size=[1, 2, 2, 1], strides=2)
	# Convolutional Layer #11
	conv11 = tf.nn.conv2d(
	   	inputs=pool4,
	   	kernel_size=[3, 3, 512, 512],
		strides=(1, 1, 1, 1),
	   	padding="SAME")
	relu11 = tf.nn.relu(conv11)
	# Convolutional Layer #12
	conv12 = tf.nn.conv2d(
	   	inputs=relu11,
	   	kernel_size=[3, 3, 512, 512],
		strides=(1, 1, 1, 1),
	   	padding="SAME")
	relu12 = tf.nn.relu(conv12)
	# Convolutional Layer #13
	conv13 = tf.nn.conv2d(
	   	inputs=relu12,
	   	kernel_size=[3, 3, 512, 512],
		strides=(1, 1, 1, 1),
	   	padding="SAME")
	relu13 = tf.nn.relu(conv13)
	# Pooling Layer #5
	pool5 = tf.layers.max_pool(inputs=relu13, pool_size=[1, 2, 2, 1], strides=2)
	#dense layer
	dense1 = tf.layers.dense(inputs=pool5, units=4096, activation=tf.nn.relu)
	dense2=tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)
	logits=tf.layers.dense(inputs=dense2, units=5, activation=tf.nn.relu)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
		labels=labels, predictions=predictions["classes"])}

	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    path='./train'
    batches, labels = get_image(path)
    print (type(batches),batches.shape)
    print (type(labels),labels.shape)
    #eval_data = mnist.test.images  # Returns np.array
    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
		model_fn=vgg_model, model_dir="./vgg16_trained_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": batch},
        y=label,
        batch_size=10,
        num_epochs=5,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": eval_data},
    #    y=eval_labels,
    #    num_epochs=1,
    #    shuffle=False)
    #eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    #print(eval_results)


if __name__ == "__main__":
    tf.app.run()
