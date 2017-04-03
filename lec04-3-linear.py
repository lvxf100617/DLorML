import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

W = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range (2001):
		cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
											feed_dict = {X: x_data, Y: y_data})
		if step % 10 == 0:
			print(step, "Cost: ", cost_val, "\nPrediction\n", hy_val)

	print("Your Score will be: ", sess.run(hypothesis, feed_dict={X: [[100,200,150]]}))

	print("Your friend score will be: ", sess.run(hypothesis, feed_dict={X: [[70, 80, 100]]}))
