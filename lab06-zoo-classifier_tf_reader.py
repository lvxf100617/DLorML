import tensorflow as tf
tf.set_random_seed(777)

filename_queue=tf.train.string_input_producer(
		['data-04-zoo.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults=[[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],
		[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = \
		tf.train.batch([xy[0:-1], xy[-1:]], batch_size=100)

#print(train_x_batch.shape, train_y_batch.shape)

nb_classes = 7

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
#print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
#print("reshape", Y_one_hot)


W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#sess.run(tf.initialize_local_variables())
#sess.run(tf.initialize_all_variables())

prediction=tf.argmax(hypothesis, 1)
correct_prediction=tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for step in range (2001):
		x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
		sess.run(optimizer, feed_dict={X: x_batch, Y: y_batch})

		if step % 200 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict={
				X: x_batch, Y: y_batch})
			print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
	
	
	coord.request_stop()
	coord.join(threads)
	
	pred = sess.run(prediction, feed_dict={X: x_batch})
	for p, y in zip(pred, y_batch.flatten()):
		print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

