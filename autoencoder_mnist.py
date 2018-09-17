from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf 

import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt
from functools import partial

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("",one_hot= True)



n_epochs = 500
batch_size=100

learning_rate=0.01

n_inputs = 28*28
n_hidden_1 = 300
n_hidden_2 = 150
n_hidden_3 = 300
n_outputs=28*28

l2 = 0.001
X = tf.placeholder(tf.float32,shape=(None,784))
def get_weights():
	return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('kernel:0')]

def get_biases():
	return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('bias:0')]


acti = tf.nn.elu
he_init = tf.contrib.layers.variance_scaling_initializer()
l2_reg = tf.contrib.layers.l2_regularizer(l2)


output_h1 = tf.layers.dense(X,n_hidden_1,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h2 = tf.layers.dense(output_h1,n_hidden_2,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h3 = tf.layers.dense(output_h2,n_hidden_3,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output = tf.layers.dense(output_h3,n_inputs,activation = None,kernel_initializer= he_init,kernel_regularizer = l2_reg)

reconstruction_loss = tf.reduce_mean(tf.square(output-X))
#reg_loss = l2_reg(get_weights()[0])+l2_reg(get_weights()[1])+l2_reg(get_weights()[2])+l2_reg4(get_weights()[3])
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add(reconstruction_loss,reg_loss)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

def plot_image(image,shape=[28,28]):
	plt.imshow(image.reshape(shape),cmap="Greys",interpolation = "nearest")
	plt.axis("off")
def plot_image_h2(image,shape=[15,10]):
	plt.imshow(image.reshape(shape),cmap="Greys",interpolation = "nearest")
	plt.axis("off")
	
	


with tf.Session() as sess:
	init.run()
	output_h2=tf.round(output_h2)
	with tf.name_scope("phase1"):
		phase1_out = tf.matmul(output_h1,get_weights()[3])+get_biases()[3]
		reconstruction_loss_p1 = tf.reduce_mean(tf.square(phase1_out-X))
		reg_loss_p1 = l2_reg(get_weights()[3])
		loss_p1 = tf.add(reconstruction_loss_p1,reg_loss_p1)
		training_op_p1 = optimizer.minimize(loss_p1)


	with tf.name_scope("phase2"):
		reconstruction_loss_p2 = tf.reduce_mean(tf.square(output_h3-output_h1))
		reg_loss_p2 = l2_reg(get_weights()[1])+l2_reg(get_weights()[2])
		loss_p2 = tf.add(reconstruction_loss_p2,reg_loss_p2)
		train_vars = [get_weights()[2],get_weights()[1],get_biases()[2],get_biases()[1]]
		training_op_p2= optimizer.minimize(loss_p2,var_list=train_vars)



	for epoch in range(n_epochs):
		X_batch,Y_batch = mnist.train.next_batch(batch_size)
		sess.run(training_op_p1,feed_dict={X:X_batch})

	for epoch in range(n_epochs):
		X_batch,Y_batch = mnist.train.next_batch(batch_size)
		sess.run(training_op_p2,feed_dict={X:X_batch})

		

	X_test = mnist.test.images[:10]
	out = sess.run(output,feed_dict={X:X_test})		
	for i in range(10):
		plt.subplot(2,10,i+1)
		plot_image(X_test[i])
		plt.subplot(2,10,i+11)
		plot_image_h2(out[i])


	print(out[2])
	plt.show()