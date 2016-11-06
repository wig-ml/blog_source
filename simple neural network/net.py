import numpy as np
import pandas as pd
import tensorflow as tf
import os, json, heapq, operator
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from bisect import bisect, bisect_left, bisect_right
from input_manager import InputManager as im

trials = 9
parameters = []
result_dict = {}
def convert(params):
    return 'S = ' + str(params[0]) + ", L = " + str(params[1])

while trials:
    trials -= 1
    
    params = (np.random.randint(1,8), np.random.randint(5,21))
    while params not in parameters:
        parameters.append(params)
        
    key = convert(params)
    result_dict[key] = {}
    
    print(key)
    im = InputManager(shift=params[0], length=params[1], reach=7)
    im.extract_data()
    im.diff()
    im.scale()
    print('extracting and preprocessing over')

    mean, std = np.mean(im.samples), np.std(im.samples)

    s = (im.samples - mean)/std

    s.shape

    C = np.random.permutation(s.shape[0])
    L = s.shape[0]

    X, Y = np.array(s[int(0.7*L):]), np.array(im.labels[int(0.7*L):])
    valx, valy = np.array(s[int(0.7*L):int(0.9*L)]), np.array(im.labels[int(0.7*L):int(0.9*L)])

    learning_rate=1e-3
    batch=100
    display=int(1e2)
    layer_size = 256
    iterations = int(4e3)

    input_size = X.shape[1]
    output_size = Y.shape[1]

    x = tf.placeholder('float', [None, input_size])
    y = tf.placeholder('float', [None, output_size])

    weights = {
        'layer1': tf.Variable(tf.random_normal([input_size, layer_size])),
        'out': tf.Variable(tf.random_normal([layer_size, output_size]))
    }

    biases = {
        'layer1': tf.Variable(tf.random_normal([layer_size])),
        'out': tf.Variable(tf.random_normal([output_size]))
    }

    def MLP(x, weigths, biases):
        hidden = tf.add(tf.matmul(x, weights['layer1']), biases['layer1'])
        hidden = tf.nn.relu(hidden)

        out = tf.add(tf.matmul(hidden, weights['out']), biases['out'])
        return out

    out = MLP(x, weights, biases)
    acc = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.initialize_all_variables()

    def get_batch(x, y, batch, iteration):
        L = x.shape[0]
        start = (batch*iteration-batch)%L
        return x[start:start+batch], y[start:start+batch]

    train_costs = []
    val_costs = []
    acc_train = []
    acc_val = []

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations):

            data, labels = get_batch(X, Y, batch, i)

            _, c = sess.run([optimizer, loss], feed_dict={x:data, y:labels})



            if i%display==0:
                ac, L = sess.run([acc,loss], feed_dict={x: valx, y:valy})
                ac_t = sess.run(acc, feed_dict={x:X, y:Y})
                train_costs.append(c)
                val_costs.append(L)
                acc_train.append(np.mean(ac_t))
                acc_val.append(np.mean(ac))
                
    result_dict[key]['train_cost'] = train_costs
    result_dict[key]['val_cost'] = val_costs
    result_dict[key]['acc_train'] = acc_train
    result_dict[key]['acc_val'] = acc_val
    
    print(key, 'finished')
    
