
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784]) # train or test data, matrix:[n,784]

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)  # matrix:[n, 10]

y_ = tf.placeholder(tf.float32, [None, 10])

# reduction_indices : 0-行之间的操作,1-列之间的操作
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))   # loss function

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

for s in range(1000):   # run computation graph
	batch_xs, batch_ys = mnist.train.next_batch(100)
	print(batch_xs.shape, batch_ys.shape)
	train_step.run({x : batch_xs, y_ : batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
print(sess.run(accuracy, {x:mnist.test.images, y_:mnist.test.labels}))
print(W.eval())
print(b.eval())