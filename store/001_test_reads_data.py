#!/usr/bin/python
'''
A Multilayer Perceptron implementation example using TensorFlow library.
'''
import tensorflow as tf

# Import data
#filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
filename_queue = tf.train.string_input_producer(["data.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0], [0], [0], [0]]
col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)
#features = tf.pack([col1, col2, col3, col4])
features = tf.pack([col1, col2])
X = tf.Variable(0)
Y = tf.Variable(0)

# Training Parameters
learning_rate = 0.001
training_epochs = 1
batch_size = 1
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_input = 361 # Go boards are 19x19 = 361
n_classes = 3 # Solving for corner, side, or center zones
n_vertices = 19 # size of board...

# I need to convert input sparse tensors to dense before training
# board is temporary accumulator where I do the conversion.
board = tf.Variable(tf.zeros([n_vertices, n_vertices]))

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
    return tf.matmul(layer_2, _weights['out']) + _biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

# Load training data
# Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #  Because I happen to know that there are 50 lines of data in the file...
    for i in range(50):
       # Retrieve a single instance:
       X, Y, val, y = sess.run([col1, col2, col3, col4])
       print "X", X
       print "Y", Y
       print "val", val
       st = tf.SparseTensor(indices=[[X, Y]], values=[val], shape=[n_vertices, n_vertices])
       board = tf.add(tf.to_int32(board), tf.sparse_tensor_to_dense(st, default_value=0, validate_indices=True, name=None))

       print "board"

       print "\nfrom data.csv"
       print "val", val
       print "y", y

    coord.request_stop()
    coord.join(threads)


    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
