import tensorflow as tf
import random
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W1 = tf.Variable(tf.random_normal([784, 500]), name='weight1')
b1 = tf.Variable(tf.random_normal([500]), name='bias1')
layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)
#layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.random_normal([500, 250]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([250]), name = 'bias2')
layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)
#layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
W3 = tf.Variable(tf.random_normal([250, nb_classes]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([nb_classes]), name = 'bias3')
#layer3 = tf.nn.softmax(tf.matmul(layer2,W3) + b3)
#layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

#W4 = tf.Variable(tf.random_normal([5, nb_classes]), name = 'weight4')
#b4 = tf.Variable(tf.random_normal([nb_classes]), name = 'bias4')
#layer4 = tf.nn.softmax(tf.matmul(layer3,W4) + b4)
#layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)
#hypothesis = tf.nn.softmax(layer4)
#W5 = tf.Variable(tf.random_normal([50, nb_classes]), name = 'weight5')
#b5 = tf.Variable(tf.random_normal([nb_classes]), name = 'bias5')
#hypothesis = tf.nn.softmax(tf.matmul(layer4, W5) + b5)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1.6).minimize(cost)


is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 10
batch_size = 100


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch=int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
			avg_cost += c / total_batch

		print('Epoch: ', '%04d' % (epoch + 1),
				'cost = ', '{:.9f}'.format(avg_cost))

	print("Learning finished")

	print("Accuracy: ", accuracy.eval(session=sess,
		feed_dict={X: mnist.test.images,
			Y: mnist.test.labels}))

	r = random.randint(0, mnist.test.num_examples - 1)
	print("Label: ", sess.run(tf.argmax(mnist.test.labels[r: r + 1], 1)))
	print("Prediction: ", sess.run(
		tf.argmax(hypothesis, 1), feed_dict = {
			X: mnist.test.images[r:r+1]}))
