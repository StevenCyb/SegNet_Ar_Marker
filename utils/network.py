import os
import cv2
import random
import numpy as np
from copy import deepcopy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils.sample_generator as sg

class Network:
    def __init__(self, shape=(512, 512, 3), classes=2, learning_rate=0.0001):
        # Set tile number and size
        self.shape = shape
        self.classes = classes
        
        # Reset old session stuff because of a recovery bug (see https://github.com/tflearn/tflearn/issues/527)
        tf.reset_default_graph()

        # Define input and output
        self.inputs = tf.placeholder(tf.float32, [None, shape[0], shape[1], shape[2]], name='inputs')
        self.outputs = tf.placeholder(tf.float32, [None, shape[0], shape[1], self.classes], name='inputs')

        # Create segnet network
        self.network = self.create_network(self.inputs)
        # Define softmax cross entropy loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.network, labels=self.outputs))
        # Define adam optimizer with lr
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        # Create tensorflow session
        self.sess = tf.Session()
        # Initialize instanced variables
        self.sess.run(tf.global_variables_initializer())

    def load_weights(self, weights_path='./weights/weights.ckpt'):
        if not os.path.isdir(os.path.dirname(weights_path)):
            raise Exception("Cannot finde weights on path '" + weights_path + "'")
        # Load the weights for the generator
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "segmentor"))
        saver.restore(self.sess, weights_path)
        print("Weights loaded")
        
    def create_network(self, inputs, dropout=0.0):
        # Create the segnet model as described in the paper
        with tf.variable_scope("segmentor", reuse=None):
            ## Encoder
            x = self.conv_block(inputs, no_convs=2, filters=64, dropout=dropout)
            skip1 = x
            x = self.conv_block(x, no_convs=2, filters=128, dropout=dropout)
            skip2 = x
            x = self.conv_block(x, no_convs=3, filters=256, dropout=dropout)
            skip3 = x
            x = self.conv_block(x, no_convs=3, filters=512, dropout=dropout)
            skip4 = x
            x = self.conv_block(x, no_convs=3, filters=512, dropout=dropout)
            ## Decoder
            x = self.conv_transpose_block(x, no_convs=3, filters=512, dropout=dropout)
            x = tf.add(x, skip4)
            x = self.conv_transpose_block(x, no_convs=3, filters=512, dropout=dropout, recude_on_last=True)
            x = tf.add(x, skip3)
            x = self.conv_transpose_block(x, no_convs=3, filters=256, dropout=dropout, recude_on_last=True)
            x = tf.add(x, skip2)
            x = self.conv_transpose_block(x, no_convs=2, filters=128, dropout=dropout, recude_on_last=True)
            x = tf.add(x, skip1)
            x = self.conv_transpose_block(x, no_convs=2, filters=64, dropout=dropout)
            ## Out
            x = slim.conv2d(x, self.classes, 1, activation_fn=None, scope='logits')
            x = slim.softmax(x)
            return x

    def conv_block(self, inputs, no_convs=1, filters=1, kernel_size=[2, 2], dropout=0.0):
        # Define i conv layers followed by relu and dropout (if not set to 0.0)
        for i in range(0, no_convs):
            inputs = slim.conv2d(inputs, filters, kernel_size, activation_fn=None, normalizer_fn=None)
            inputs = tf.nn.relu(slim.batch_norm(inputs))
            if dropout != 0.0:
                inputs = slim.dropout(inputs, keep_prob=dropout)
        # Define pooling after conv block
        inputs = slim.pool(inputs, [2, 2], stride=[2, 2], pooling_type='MAX')
        return inputs

    def conv_transpose_block(self, inputs, no_convs=1, filters=1, kernel_size=[2, 2], stride=[2, 2], dropout=0.0, recude_on_last=False):
        no_convs -= 1
        # Define conv transpose to scale up
        inputs = slim.conv2d_transpose(inputs, filters, kernel_size=kernel_size, stride=stride, activation_fn=None)
        # Define i-1 conv layers followed by relu and dropout (if not set to 0.0)
        for i in range(0, no_convs):
            if recude_on_last and (i + 1) == no_convs:
                inputs = slim.conv2d(inputs, int(filters / 2), kernel_size, activation_fn=None, normalizer_fn=None)
            else:
                inputs = slim.conv2d(inputs, filters, kernel_size, activation_fn=None, normalizer_fn=None)
            inputs = tf.nn.relu(slim.batch_norm(inputs))
            if dropout != 0.0:
                inputs = slim.dropout(inputs, keep_prob=dropout)
        return inputs

    def one_hot_it(self, mask):
        # Transform mask into one-hot-format
        # See implementation of  https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
        mask = np.float32(mask) / 255.0
        label_values = []
        label_values.append([1.0, 1.0, 1.0]) # Marker label
        label_values.append([0.0, 0.0, 0.0]) # Background label
        semantic_map = []
        for colour in label_values:
            equality = np.equal(mask, colour)
            class_map = np.all(equality, axis = -1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1)
        return np.float32(semantic_map)
    
    def reverse_one_hot(self, mask):
        # Transform mask from one-hot-format back to image format
        # See implementation of  https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
        mask = np.argmax(mask, axis = -1)
        label_values = []
        label_values.append([1, 1, 1]) # Marker label
        label_values.append([0, 0, 0]) # Background label
        colour_codes = np.array(label_values)
        mask = colour_codes[np.around(mask, decimals=0).astype(int)]
        return np.uint8(mask * 255)
    
    def train_synthetic_data(self, epochs=5000, steps=100, batch_size=1, weights_path='./weights/weights.ckpt', saving_epochs=500, DICT_4X4_50=False, DICT_4X4_100=False, DICT_4X4_250=False, DICT_4X4_1000=False, DICT_5X5_50=False, DICT_5X5_100=False, DICT_5X5_250=False, DICT_5X5_1000=False, DICT_6X6_50=False, DICT_6X6_100=False, DICT_6X6_250=False, DICT_6X6_1000=False, DICT_7X7_50=False, DICT_7X7_100=False, DICT_7X7_250=False, DICT_7X7_1000=False, DICT_ARUCO_ORIGINAL=False, DICT_APRILTAG_16h5=False, DICT_APRILTAG_25h9=False, DICT_APRILTAG_36h10=False, DICT_APRILTAG_36h11=True):
        saver = tf.train.Saver()
        # Generate one sample to calculate accuracy
        image, mask = sg.generate_random_full_synthetic_sample(self.shape, DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL, DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, DICT_APRILTAG_36h10, DICT_APRILTAG_36h11)
        callback_image = np.float32(image)/255.0
        callback_mask = self.one_hot_it(mask)
        # Training loop
        for epoch in range(0, epochs + 1):
            # Create two batches for input images and a batch with the ground truth
            input_batch = np.zeros([batch_size, self.shape[0], self.shape[1], self.shape[2]])
            output_batch = np.zeros([batch_size, self.shape[0], self.shape[1], self.classes])
            # Generate the batch
            for batch_index in range(batch_size):
                image = None
                mask = None
                # Switch between full synthetic and half synthetic samples
                if epoch % 2 == 0:
                    image, mask = sg.generate_random_full_synthetic_sample(self.shape, DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL, DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, DICT_APRILTAG_36h10, DICT_APRILTAG_36h11)
                else:
                    image, mask = sg.generate_random_half_synthetic_sample(self.shape, DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL, DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, DICT_APRILTAG_36h10, DICT_APRILTAG_36h11)
                input_batch[batch_index] = np.float32(image)/255.0
                output_batch[batch_index] = self.one_hot_it(mask)
            # Run the training n times where n = steps
            for step in range(steps):
                _, current_loss = self.sess.run([self.optimizer, self.loss],
                    feed_dict={self.inputs: input_batch, self.outputs: output_batch})
                # Do prediction on sample and calculate accuracy
                input_image = np.zeros([1, self.shape[0], self.shape[1], self.shape[2]])
                input_image[0, :, :, :] = deepcopy(callback_image)
                mask = self.sess.run(self.network, feed_dict={self.inputs: input_image})
                accuracy = self.compute_accuracy(mask, deepcopy(callback_mask))
                # Print training information
                print("Epoch: %d/%d, Loss: %g, Accuracy: %g" % (epoch, epochs, current_loss, accuracy))
                # Save weights if triggered
                if epoch % saving_epochs == 0:
                    print("Save weights")
                    if not os.path.isdir("weights"):
                        os.mkdir("weights")
                    saver.save(self.sess, weights_path)
    
    def read_resize_image(self, path):
        return cv2.resize(cv2.imread(path), (self.shape[0], self.shape[1]))
    
    def train_real_data(self, epochs=5000, steps=100, batch_size=1, weights_path='./weights/weights.ckpt', saving_epochs=500, dataset=None):
        if dataset is not None:
            saver = tf.train.Saver()
            # Training loop
            for epoch in range(0, epochs + 1):
                # Create two batches for input images and a batch with the ground truth
                input_batch = np.zeros([batch_size, self.shape[0], self.shape[1], self.shape[2]])
                output_batch = np.zeros([batch_size, self.shape[0], self.shape[1], self.classes])
                # Generate the batch
                for batch_index in range(batch_size):
                    sample = random.choice(dataset)
                    input_batch[batch_index] = np.float32(self.read_resize_image(sample[0]))/255.0
                    output_batch[batch_index] = self.one_hot_it(self.read_resize_image(sample[1]))
                # Run the training n times where n = steps
                for step in range(steps):
                    _, current_loss = self.sess.run([self.optimizer, self.loss],
                                                    feed_dict={self.inputs: input_batch, self.outputs: output_batch})
                # Do prediction on sample and calculate accuracy
                input_image = np.zeros([1, self.shape[0], self.shape[1], self.shape[2]])
                input_image[0, :, :, :] = self.read_resize_image(dataset[0][0])
                mask = self.sess.run(self.network, feed_dict={self.inputs: input_image})
                accuracy = self.compute_accuracy(mask, self.one_hot_it(self.read_resize_image(dataset[0][1])))
                # Print training information
                print("Epoch: %d/%d, Loss: %g, Accuracy: %g" % (epoch, epochs - 1, current_loss, accuracy))
                # Save weights if triggered
                if epoch % saving_epochs == 0:
                    print("Save weights")
                    if not os.path.isdir("weights"):
                        os.mkdir("weights")
                    saver.save(self.sess, weights_path)

    def compute_accuracy(self, mask, label):
        # Finde absolute difference and calculate the accuracy
        count = 0.0
        mask = mask.flatten()
        label = label.flatten()
        for i in range(len(label)):
            if mask[i] == label[i]:
                count = count + 1.0
        return float(count) / float(len(label))
    
    def predict(self, image=None):
        # Save the original size and resize to network input size
        height, width, channels = image.shape
        image = cv2.resize(image, (self.shape[0], self.shape[1]))
        # Create an input batch and set image
        input_image = np.zeros([1, self.shape[0], self.shape[1], self.shape[2]])
        input_image[0, :, :, :] = image
        # Run prediction
        image = self.sess.run(self.network, feed_dict={self.inputs: input_image})
        # Reverse the one-hot-output
        image = self.reverse_one_hot(np.array(image[0,:,:,:]))
        # Resize back to original input size
        image = cv2.resize(image, (width, height))
        return image
