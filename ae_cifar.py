from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt
from functools import partial


import cifar10
#from tensorflow.examples.tutorials.cifar10 import input_data

#mnist = input_data.read_data_sets("C:\Users\vsaik\Downloads\Compressed\cifar-10-batches-py",one_hot= True)

#cifar10.maybe_download_and_extract()
from cifar10 import img_size, num_channels, num_classes

n_epochs = 100
batch_size=250
n_batches = 50000 // batch_size

learning_rate=0.00001

n_inputs = 3072
n_hidden_1 = 1500
n_hidden_2 = 750
n_hidden_3 = 1500
n_outputs=3072

l2 = 0.001
X = tf.placeholder(tf.float32,shape=(None,n_inputs))




def get_weights():
	return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('kernel:0')]

def get_biases():
	return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('bias:0')]


def plot_images(images, cls_true, cls_pred=None, smooth=True):

	assert len(images) == len(cls_true) == 9

	# Create figure with sub-plots.
	fig, axes = plt.subplots(3, 3)

	# Adjust vertical spacing if we need to print ensemble and best-net.
	if cls_pred is None:
		hspace = 0.3
	else:
		hspace = 0.6
	fig.subplots_adjust(hspace=hspace, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Interpolation type.
		if smooth:
			interpolation = 'spline16'
		else:
			interpolation = 'nearest'

		# Plot image.
		ax.imshow(images[i, :, :, :],
				  interpolation=interpolation)
			
		# Name of the true class.
		cls_true_name = class_names[cls_true[i]]

		# Show true and predicted classes.
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true_name)
		else:
			# Name of the predicted class.
			cls_pred_name = class_names[cls_pred[i]]

			xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

		# Show the classes as the label on the x-axis.
		ax.set_xlabel(xlabel)
		
		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])
	
	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	plt.show()





def plot_image(image,shape=[32,32,3]):
	plt.imshow(image.reshape(shape),cmap="Greys",interpolation = "nearest")
	plt.axis("off")

def plot_image_h2(image,shape=[25,10,3]):
	plt.imshow(image.reshape(shape),cmap="Greys",interpolation = "nearest")
	plt.axis("off")
	
	
	
		

images_train, cls_train, labels_train = cifar10.load_training_data()

images_test, cls_test, labels_test = cifar10.load_test_data()




acti = tf.nn.elu
he_init = tf.contrib.layers.variance_scaling_initializer()
l2_reg = tf.contrib.layers.l2_regularizer(l2)


output_h1 = tf.layers.dense(X,n_hidden_1,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h2 = tf.layers.dense(output_h1,n_hidden_2,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h3 = tf.layers.dense(output_h2,n_hidden_3,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output = tf.layers.dense(output_h3,n_inputs,activation = None,kernel_initializer= he_init,kernel_regularizer = l2_reg)
outputs = tf.sigmoid(output)

reconstruction_loss = tf.reduce_mean(tf.square(output-X))
#reg_loss = l2_reg(get_weights()[0])+l2_reg(get_weights()[1])+l2_reg(get_weights()[2])+l2_reg4(get_weights()[3])
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add(reconstruction_loss,reg_loss)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()



with tf.Session() as sess:
	init.run()
	output_h2 = tf.round(output_h2)
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




	print("shape:",np.shape(images_train[49999]))
	
	
	for epoch in range(n_epochs):
		for i in range(n_batches):
			X_batch = []
			for j in range(i*batch_size,(i+1)*batch_size):
				if j > 49999:
					j = j % 50000
				img = images_train[j].reshape([3072])
				X_batch.append(img)
			sess.run(training_op_p1,feed_dict={X:X_batch})

	for epoch in range(n_epochs):
		for i in range(n_batches):
			X_batch = []
			for j in range(i*batch_size,(i+1)*batch_size):
				img = images_train[j].reshape([3072])
				X_batch.append(img)
			sess.run(training_op_p2,feed_dict={X:X_batch})


	reconstruction_loss = reconstruction_loss_p2 + reconstruction_loss_p1
	X_test = []
	for j in range(3):
		img = images_test[j].reshape([3072])
		X_test.append(img)
	out = sess.run(output,feed_dict={X:X_test})	
	
	for i in range(3):
		plt.subplot(3,2,i*2+1)
		plot_image(images_test[i])
		plt.subplot(3,2,i*2+2)
		plot_image(out[i])
	plt.show()
