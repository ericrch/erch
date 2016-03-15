#!/usr/bin/python
'''
A Multilayer Perceptron implementation example using TensorFlow library.
'''
import tensorflow as tf
import numpy as np

###############
# Import data #
###############
#filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
filename_queue = tf.train.string_input_producer(["data.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0], [0], [0], [0]]
col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)


#######################
# Training Parameters #
#######################
learning_rate = 0.001
training_epochs = 10
batch_size = 50 # I happen to know there are 50 items in the current training file...
display_step = 1

#############################
# Neural network Parameters #
#############################
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_input = 361 # Go boards are 19x19 = 361
n_classes = 1 # Solving for corner, side, or center zones
n_vertices = 19 # size of board...

#############################
#     training data         #
#############################
# I need to convert input sparse tensors to dense before training
X_vertex = tf.Variable(0) #  X position input comes from training data
Y_vertex = tf.Variable(0) #  Y position input comes from training data
board = tf.Variable(tf.zeros([n_vertices, n_vertices])) # accumulator where I convert from sparse to dense tensor.
batch_x = [n_input]       # this is what get's fed in as training patterns
target_y = []             # this is what the result of each pattern should look like!


#############################
#   TF Graph input          #
#############################
x = tf.placeholder(tf.int32, shape=(1, n_input))  #  batch_x fed into this 
#y = tf.placeholder(tf.int32, shape=(1, n_classes))# target_y fed into this
y = tf.placeholder(tf.int32, shape=(n_classes))# target_y fed into this

##########################################################################
"""
# Create model
def multilayer_perceptron(_x, _weights, _biases):
    print "\n\n_x", _x
    print "_weights", _weights
    print "_biases", _biases
    layer_1 = tf.nn.relu(tf.add(tf.matmul(tf.to_float(_x), _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    print "layer_1", layer_1
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
    print "layer_2", layer_2
    print "_weights", tf.shape(_weights)
    print "_biases", tf.shape(_biases)
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
"""
##########################################################################

# Create model
def perceptron(_x, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(tf.to_float(_x), _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    print "\n\n_x", _x
    print "layer_1", layer_1
    store = tf.nn.relu(tf.matmul(layer_1, _weights['out']) + _biases['out'])
    print "store ", store
    return store

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
#pred = multilayer_perceptron(x, weights, biases)
pred = perceptron(x, weights, biases)
print "pred ", pred

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, tf.to_float(tf.expand_dims(y,0)))) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    ######################
    # Load training data #
    ######################
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #  Because I happen to know that there are batch_size lines of data in the file...
    for epoch in range(batch_size):
       #print "\nfrom data.csv"

       # Retrieve a single instance.  In this case, X_vertex and Y_vertex are coordinates of vertices on the board.
       # val is the color of the stone.
       # target_y is the target output.
       X_vertex, Y_vertex, val, temp_y = sess.run([col1, col2, col3, col4])
       print "\n\nX_vertex", X_vertex
       print "Y_vertex", Y_vertex
       print "val", val
       print "temp_y", temp_y

       st = tf.SparseTensor(indices=[[X_vertex -1, Y_vertex -1]], values=[val], shape=[n_vertices, n_vertices])
       board = tf.add(tf.to_int32(board), tf.sparse_tensor_to_dense(st, default_value=0, validate_indices=True))
       #print "board shape", tf.shape(board)
       print sess.run(board)

       print "\n\ny", y
       target_y.insert(0, temp_y)
       print "target_y", target_y
       print "target_y shape", tf.shape(target_y)

       print "\nx", x
       batch_x = (tf.expand_dims(tf.reshape(board, [-1]),0)).eval()
       #print "batch_x ", batch_x

       print "\n\n"

       ######################
       # Training cycle
       ######################
       avg_cost = 0.
       sess.run(optimizer, feed_dict={x: batch_x, y: target_y})   ###  <---------Now it wants a tensor!!!

       # Compute average loss
       #avg_cost += sess.run(cost, feed_dict={x: x, y: y})
       
       # Clean up target_y
       del target_y[0]

       # Display logs per epoch step
       if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

    coord.request_stop()
    coord.join(threads)

