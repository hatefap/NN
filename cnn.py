#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf



import numpy as np
import sys
import matplotlib.pyplot as plt
import math




import glob




#directory location of cifar-10 binary files
CIFAR_LOCATTION = '/home/hatef/cifar-10/cifar-10-batches-bin/'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCH = 50




# produce images for training from data_batch_i where 0 < i < 6
def image_training_generator():
    # if you look at cifar-10 extracted zip file, you see following pattern among training files  
    common_name = 'data_batch_?.bin'
    bin_files = []
    for file in glob.glob(CIFAR_LOCATTION + common_name):
        bin_files.append(file)
    
    while(len(bin_files) > 0):
        file_path = bin_files.pop()
        with open(file_path, 'rb') as handle:
            while True:
                label = handle.read(1) # first byte is label of image, label is between 0-9
                image = handle.read(3072) # next 3072 bytes are image itself
                # if we reach eof file
                if len(label) < 1 or len(image) < 3072: 
                    break
                label = int.from_bytes(label, byteorder=sys.byteorder)
                one_hot_encoded_label = np.zeros(shape=(10,))
                one_hot_encoded_label[label] = 1
                
                image = np.array([b for b in image])
                R = image[0:1024].reshape(32,32)
                G = image[1024:2048].reshape(32,32)
                B = image[2048:].reshape(32,32)
                yield np.dstack((R, G, B)) / 255, one_hot_encoded_label
                




# you can use similar approach to generate test data, but here we want to use another common approach
test_pickle_python = '/home/hatef/cifar-10/python/cifar-10-batches-py/test_batch'
def unpickle(file_location):
    import _pickle as cPickle
    with open(file_location, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

test_data = unpickle(test_pickle_python)




x_test = test_data['data']
y_test = test_data['labels']

x_test = x_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
x_test = x_test / 255

n_classes = 10
y_test = np.array(y_test).reshape(-1)
y_test = np.eye(n_classes)[y_test]




# if you have variance problem use this class to increase your training data
class ImageTransformation:
    @staticmethod
    def flip_image_horizental(images):
        # images --> [batch, height, width, channels]
        return tf.image.flip_left_right(images)
    
    @staticmethod
    def flip_image_vertical(images):
        # images --> [batch, height, width, channels]
        return tf.image.flip_up_down(images)
    
    @staticmethod
    def rotate_image(images, degree):
        return tf.contrib.image.rotate(images, math.radians(degree))
    
    @staticmethod
    def transform_image(images, transform_vector):
        return tf.contrib.image.transform(images, transform_vector)



# we use 2 convolutional layers, and use max pool in each convolutional layer, our classification layer 
# is 2 dense layers 
def do_conv(input_tensor, W):
    # input_tensor --> [batch, height of image, width of image, number of channels]
    # W --> W is our kernel, [filter height, filter width, input channel, output channel].
    
    # padding SAME retain size of image during convolution operation by adding zero padding to image
    return tf.nn.conv2d(input_tensor, filter=W, padding='SAME', strides=[1,1,1,1])


def do_max_pool(input_tensor, window=2):
    # we do subsampling with this method i.e. if window=2 then tensor [10, 32, 32, 64] becomes [10, 16, 16, 64] after this operation
    return tf.nn.max_pool(input_tensor, ksize=(1, window, window, 1), strides=(1, window, window, 1), padding='VALID')

def dense_layer(input_tensor, neurons=128):
    # input_tensor --> [N(batch), M]
    dense_layer_weight = tf.Variable(tf.truncated_normal(shape=[int(input_tensor.get_shape()[1]), neurons], mean=0.0, stddev=0.2))
    dense_layer_bias = tf.Variable(tf.truncated_normal(shape=[neurons], mean=0.0, stddev=0.2))
    
    return tf.add(tf.matmul(input_tensor, dense_layer_weight), dense_layer_bias)
    




X = tf.placeholder(dtype=tf.float32, shape=(None, 32,32,3), name='input_images')

# none is batch size and 10 is the number of classes in encoded form
y_true = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='label_of_input_images')




# we can use at most one -1 in shape argument. because the number of batch is unknown until run time. we use -1 
# and tensor flow sets appropriate number in run time
net_in = tf.reshape(X, shape=[-1, 32,32,3])

# our first convolution layer

filter_1 = tf.Variable(tf.truncated_normal(shape=[2, 2, 3, 32], stddev=0.5))
bias_filter_1 = tf.Variable(tf.truncated_normal([int(filter_1.get_shape()[3])]))
conv_1 = tf.nn.bias_add(do_conv(net_in, filter_1), bias_filter_1)


# pass through elu activation function
result_conv_1 = tf.nn.elu(conv_1)

# first subsample
conv_1_subsample = do_max_pool(result_conv_1)

# our second convolution layer
filter_2 = tf.Variable(tf.truncated_normal(shape=[2, 2, 32, 32], stddev=0.5))
bias_filter_2 = tf.Variable(tf.truncated_normal([int(filter_1.get_shape()[3])]))
conv_2 = tf.nn.bias_add(do_conv(conv_1_subsample, filter_2), bias_filter_2)

result_conv_2 = tf.nn.elu(conv_2)

conv_2_subsample = do_max_pool(result_conv_2)
# after two subsample our images size is 8 * 8
flat_conv_2_subsample = tf.reshape(conv_2_subsample, shape=[-1, 8*8*32])
dense_layer_1 = tf.nn.elu(dense_layer(flat_conv_2_subsample, neurons=32))
dropped_1 = tf.nn.dropout(dense_layer_1,keep_prob=0.8)
dense_layer_2 = tf.nn.elu(dense_layer(dense_layer_1, neurons=16))
dropped_2 = tf.nn.dropout(dense_layer_2,keep_prob=0.8)

classification_layer = dense_layer(dropped_2, neurons=10)




classification_layer.get_shape()




train_dataset = tf.data.Dataset.from_generator(image_training_generator, output_types=(tf.float32, tf.int32))




# using batch(BATCH_SIZE) not works!, why?
train_dataset = train_dataset.apply(tf.contrib.data.sliding_window_batch(window_size=BATCH_SIZE, window_shift=BATCH_SIZE))
train_dataset = train_dataset.repeat(EPOCH)




iterator = train_dataset.make_one_shot_iterator()
elements = iterator.get_next()





loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=classification_layer))
optimzer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimzer.minimize(loss)




vars_init = tf.global_variables_initializer()




with tf.Session() as sess:
    sess.run(vars_init)
    step = 0
    counter = 0
    while True:
        try:
            images,labels = sess.run(elements)
            sess.run(train, feed_dict={net_in:images, y_true:labels})
            step += 1
            if step%100 == 0:
                counter += 1
                preds = tf.equal(tf.argmax(classification_layer, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(preds, tf.float32))
                res = sess.run(acc, feed_dict={net_in:x_test, y_true:y_test})
                print(f'accuracy after {counter*step} updates in weights: {res}')
                step = 0
        except tf.errors.OutOfRangeError: 
            break



# accuracy after 31200 updates in weights: 0.6800000254260437







# In[ ]:




