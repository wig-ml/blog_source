import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
%matplotlib inline

X = np.random.randn(10000, 5)*np.random.randint(-3,5,10000)[None].T
Y = np.std(X, axis=1)[None].T

input_size = X.shape[1]
layer = 100
lr = 1e-2
batch = 200
iterations = int(1e4)


weights = {
    'layer': tf.Variable(tf.truncated_normal([input_size, layer], stddev=1/np.sqrt(input_size))),
    'out': tf.Variable(tf.truncated_normal([layer, 1], stddev=1/np.sqrt(input_size)))
}

biases = {
    'layer': tf.Variable(tf.zeros([layer])),
    'out': tf.Variable(tf.zeros([1]))
}

x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float', [None, 1])

def MLP(x, weights, biases):
    hidden = tf.add(tf.matmul(x, weights['layer']), biases['layer'])
    hidden = tf.nn.relu(hidden)
    
    out = tf.add(tf.matmul(hidden, weights['out']), biases['out'])
    return out

out = MLP(x, weights, biases)

loss = tf.reduce_mean(tf.square(tf.sub(out, y)))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.initialize_all_variables()

losses = []
valloss = []
C = np.random.permutation(10000)
k = 7000
tr_s, tr_l, val_s, val_l = X[C[:k]], Y[C[:k]], X[C[k:]], Y[C[k:]]

def get_batch(x, y, batch, iteration):
        L = x.shape[0]
        start = (batch*iteration-batch)%L
        return x[start:start+batch], y[start:start+batch]

with tf.Session() as sess:
    
    sess.run(init)
    
    while iterations:
        
        iterations -= 1
        
        data, labels = get_batch(tr_s, tr_l, batch, iterations)
        _, c = sess.run([optimizer, loss], feed_dict={x:data, y:labels})
        
        if iterations % 1000 == 0:
            print('left', iterations)
            losses.append(c)
            lc = sess.run(loss, feed_dict={x:val_s[:500], y:val_l[:500]})
            valloss.append(lc)
            
            
    test = sess.run(out, feed_dict={x:val_s[:500]})
        
        
        

np.linalg.norm(test - val_l[:500])/500

plt.plot(range(10), losses, color='red')
plt.plot(range(10), valloss, color='black')
plt.show()
