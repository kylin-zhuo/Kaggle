import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd
from sklearn.model_selection import train_test_split

CKPT_PATH = './cnn/cnn.ckpt'

def load_data():
    labeled_images = pd.read_csv('./data/train.csv')
    images = labeled_images.iloc[:, 1:]
    labels = labeled_images.iloc[:, :1]
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.7, random_state=0)

    # test_images[test_images>0] = 1
    # train_images[train_images>0] = 1

    test_images = test_images / 50
    train_images = train_images / 50

    train_images = shape_image(train_images)
    test_images = shape_image(test_images)

    train_images = np.array(train_images)
    test_images = np.array(test_images)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train_labels = convert_to_one_hot(train_labels, 10).T
    test_labels = convert_to_one_hot(test_labels, 10).T

    X_train = train_images.astype("float")
    Y_train = train_labels.astype("float")
    X_test = test_images.astype("float")
    Y_test = test_labels.astype("float")

    return X_train, Y_train, X_test, Y_test


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def shape_image(img):
    return np.array(img).reshape(img.shape[0], 28, 28, 1)


def create_placeholders(n_w, n_h, n_c, n_classes):
    X = tf.placeholder(dtype=tf.float32, shape=(None, n_w, n_h, n_c), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(None, n_classes), name='Y')
    return X, Y


def initialize_parameters():
    tf.set_random_seed(1)                           
    initializer = tf.contrib.layers.xavier_initializer(seed=0)
    W1 = tf.get_variable("W1", [4,4,1,16], initializer=initializer)
    W2 = tf.get_variable("W2", [2,2,16,32], initializer=initializer)
    W3 = tf.get_variable("W3", [4,4,32,64], initializer=initializer)
    return W1, W2, W3


def forward_propagation(X, W1, W2, W3):
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding='VALID')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    P3 = tf.contrib.layers.flatten(P3)
    Z4 = tf.contrib.layers.fully_connected(P3, 10, activation_fn=None)
    return Z4


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]               
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y)) 
    return cost




# def forward_propagation_for_predict(X, parameters):
#     """
#     Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
#     Arguments:
#     X -- input dataset placeholder, of shape (input size, number of examples)
#     parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
#                   the shapes are given in initialize_parameters

#     Returns:
#     Z3 -- the output of the last LINEAR unit
#     """
    
#     # Retrieve the parameters from the dictionary "parameters" 
#     W1 = parameters['W1']
#     b1 = parameters['b1']
#     W2 = parameters['W2']
#     b2 = parameters['b2']
#     W3 = parameters['W3']
#     b3 = parameters['b3'] 
#                                                            # Numpy Equivalents:
#     Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
#     A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
#     Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
#     A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
#     Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
#     return Z3

# def predict(X, parameters):
    
#     W1 = tf.convert_to_tensor(parameters["W1"])
#     b1 = tf.convert_to_tensor(parameters["b1"])
#     W2 = tf.convert_to_tensor(parameters["W2"])
#     b2 = tf.convert_to_tensor(parameters["b2"])
#     W3 = tf.convert_to_tensor(parameters["W3"])
#     b3 = tf.convert_to_tensor(parameters["b3"])
    
#     params = {"W1": W1,
#               "b1": b1,
#               "W2": W2,
#               "b2": b2,
#               "W3": W3,
#               "b3": b3}
    
#     x = tf.placeholder("float", [12288, 1])
    
#     z3 = forward_propagation_for_predict(x, params)
#     p = tf.argmax(z3)
    
#     sess = tf.Session()
#     prediction = sess.run(p, feed_dict = {x: X})
        
#     return prediction
