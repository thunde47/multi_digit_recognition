from __future__ import print_function
from six.moves import cPickle as pickle
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import os.path
import numpy as np

import matplotlib.pyplot as plt

def vertical_display(value):
	print(value)

def evolve(restore, images_used):		
	iterations=100
	batch_size = 16
	patch_size = 3
	depth = 32
	num_hidden = 500
	num_channels=3
	width=64
	height=32

	achaar_file='../data_SVHN/svhn_'+str(width)+'x'+str(height)+'x'+str(images_used)+'.achaar'
	print("Loading pickeled file...\n")
	with open(achaar_file,'rb') as f:
		save=pickle.load(f)
		all_labels=save['all_labels']
		if not restore:
			print("Un-achaarifying the training data from %s ...\n"%achaar_file)
			train_dataset=save['train_dataset']
			train_target=save['train_target']		
		else:
			print("Un-achaarifying the test data from %s ...\n"%achaar_file)
			test_dataset=save['test_dataset']
			test_target=save['test_target']
		del save

	print("Generating graph...\n")
	graph=tf.Graph()
	with graph.as_default():
	
		for target_number in range(1,len(all_labels)-4):
		
			num_labels=all_labels[target_number]
			tf_train_dataset=tf.placeholder(tf.float32, shape=(None, height, width, num_channels))
			tf_train_labels=tf.placeholder(tf.float32, shape=(None, num_labels))

			#Defining layers, weights and biases
			layer1_weights=tf.Variable(tf.truncated_normal([patch_size,patch_size, num_channels, depth], stddev=0.01))
			#layer1_biases=tf.Variable(tf.zeros([depth]))
			layer1_biases=tf.Variable(tf.zeros([depth]))

			layer2_weights=tf.Variable(tf.truncated_normal([patch_size, patch_size,depth,depth],stddev=0.01))
			layer2_biases=tf.Variable(tf.constant(1.0, shape=[depth]))

			layer3_weights=tf.Variable(tf.truncated_normal([width//4*height//4*depth,num_hidden],stddev=0.01))
			layer3_biases=tf.Variable(tf.constant(1.0, shape=[num_hidden]))
	
			layer4_weights=tf.Variable(tf.truncated_normal([num_hidden,num_labels],stddev=0.1))
			layer4_biases=tf.Variable(tf.constant(1.0, shape=[num_labels]))
	
			keep_prob=tf.placeholder(tf.float32)
			def model(data):
				conv1 = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
				hidden1 = tf.nn.relu(conv1 + layer1_biases)
				hidden_pool1=tf.nn.max_pool(hidden1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
				#print("conv layer 1 - ",hidden_pool1.get_shape().as_list())
				conv2 = tf.nn.conv2d(hidden_pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
				hidden2 = tf.nn.relu(conv2 + layer2_biases)
				hidden_pool2=tf.nn.max_pool(hidden2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
				#shape = hidden.get_shape().as_list()
				shape=tf.shape(hidden_pool2)
				#print("conv layer 2 - ",hidden_pool2.get_shape().as_list())
				reshape = tf.reshape(hidden_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
				#hidden = tf.nn.dropout(tf.tanh(tf.matmul(reshape, layer3_weights) + layer3_biases),keep_prob)
				hidden=	tf.nn.tanh(tf.matmul(reshape, layer3_weights) + layer3_biases)
				#print("fully connected layer",hidden.get_shape().as_list())
				return tf.matmul(hidden, layer4_weights) + layer4_biases

			logits=model(tf_train_dataset)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	
			global_step=tf.Variable(0)
			learning_rate=tf.train.exponential_decay(.1,global_step,1000,.9,staircase=True)
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
			#optimizer=tf.train.AdagradOptimizer(learning_rate=.05,initial_accumulator_value=0.1,use_locking=False)
			optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
			train_prediction=tf.nn.softmax(logits)
			correct_prediction=tf.equal(tf.argmax(train_prediction,1),tf.argmax(tf_train_labels,1))
			accuracy=100.0*tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		
		
			with tf.Session(graph=graph) as session:
				saver=tf.train.Saver()	
				#if not os.path.exists("parameters.ckpt"):
				if not restore:
					print("Training network...\n")
					tf.initialize_all_variables().run()
					train_labels=train_target[target_number]
					plot_accuracy=[]
					for step in range(iterations):
					
						offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
						batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
						batch_labels = train_labels[offset:(offset + batch_size), :]
										
						feed_dict_train = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.5}
						feed_dict_train_eval={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}
						#feed_dict_test = {tf_train_dataset : test_dataset, tf_train_labels : test_labels, keep_prob:1.0}
						_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict_train)
						batch_accuracy=accuracy.eval(feed_dict=feed_dict_train_eval)
						plot_accuracy.append(batch_accuracy)
						if (step % 100 == 0):
							print('Minibatch loss at step %d: %f' %(step, l))
							print('Minibatch accuracy: %.1f%%' %batch_accuracy)		
					
					save_path = saver.save(session, "parameters_"+str(target_number)+".ckpt")
					print("Model saved in file: %s" % save_path)
					#print('Test accuracy: %.1f%%' % accuracy.eval(feed_dict=feed_dict_test))
					plt.plot(plot_accuracy)
					plt.ylabel("Minibatch accuracy")
					plt.show()
				else:
					print("Using trained parameters for prediction...\n")
					new_saver=tf.train.import_meta_graph("parameters_"+str(target_number)+".ckpt.meta")
					new_saver.restore(session, tf.train.latest_checkpoint('./'))
					
					test_labels=test_target[target_number]
				
					feed_dict_test = {tf_train_dataset : test_dataset, tf_train_labels : test_labels, keep_prob:1.0}
				
					test_prediction=session.run([train_prediction],feed_dict=feed_dict_test)
					
					'''
					print("Originals",test_labels)
					#print(tf.nn.softmax(test_prediction).eval())	
					
					if (target_number>0):
					
						original=np.concatenate((np.array(range(all_labels[target_number]-1)),np.array([10])))
					
					else:
						original=np.array(range(1,all_labels[target_number]+1))
					print(original)	
					original_array=np.tile(original,(len(test_labels),1))
					print("Actual output, Prediction output")
					map(vertical_display,zip(np.amax(np.multiply(original,test_labels),axis=1),original_array[np.arange(len(test_prediction)),np.argmax(test_prediction[0],axis=1)]))
					#print(test_labels)
					#print(test_prediction[0].argmax(axis=1))
					#print(np.amax(np.multiply(test_prediction,test_labels)))
			
					#print('Test prediction:', test_prediction[0][5], test_labels[0])	
					'''
					print('Test accuracy: %.1f%%' % accuracy.eval(feed_dict=feed_dict_test))




