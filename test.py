#!/usr/bin/python
'''
A Multilayer Perceptron implementation example using TensorFlow library.
'''
import tensorflow as tf
import numpy as np

###############
# Import data #
###############
filename_queue = tf.train.string_input_producer(["data.csv", "data1.csv"], shuffle=False, capacity=1)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the dtype of the decoded result.
record_defaults = [[0], [0], [0], [0]]
col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)


#######################
# Training Parameters #
#######################
learning_rate = 0.01
training_epochs = 3
batch_size = 50 # I happen to know there are 50 moves per game in the current training file...
display_step = 10

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
y = tf.placeholder(tf.int32, shape=(n_classes))# target_y fed into this


#############################
#        Create model       #
#############################
def perceptron(_x, _weights, _biases):
    #layer_1 = tf.nn.relu(tf.add(tf.matmul(tf.to_float(_x), _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    layer_1 = tf.sigmoid(tf.add(tf.matmul(tf.to_float(_x), _weights['h1']), _biases['b1']))  #Hidden layer with RELU activation
    #store = tf.nn.relu(tf.matmul(layer_1, _weights['out']) + _biases['out'])
    store = tf.sigmoid(tf.matmul(layer_1, _weights['out']) + _biases['out'])
    print "store ", store
    return store

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0.0, stddev=3)),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], mean=0.0, stddev=3))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
prediction = perceptron(x, weights, biases)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, tf.to_float(tf.expand_dims(y,0)))) # Softmax loss
cost = -tf.reduce_sum((tf.log(prediction) * tf.to_float(tf.expand_dims(y,0))))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

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

    for epoch in range(training_epochs * batch_size):

       # Retrieve a single instance.  In this case, X_vertex and Y_vertex are coordinates of vertices on the board.
       # val is the color of the stone.
       # target_y is the target output.
       X_vertex, Y_vertex, val, temp_y = sess.run([col1, col2, col3, col4])
       print "\n\n----------------------------------------------------"
       print "EPOCH", epoch
       print "X_vertex=", X_vertex, "  Y_vertex=", Y_vertex, "  Stone Color", val

       st = tf.SparseTensor(indices=[[X_vertex -1, Y_vertex -1]], values=[val], shape=[n_vertices, n_vertices])
       board = tf.add(tf.to_int32(board), tf.sparse_tensor_to_dense(st, default_value=0, validate_indices=True))
       #print board.eval()
       target_y.insert(0, temp_y)

       batch_x = (tf.expand_dims(tf.reshape(board, [-1]),0)).eval()

       ######################
       # Training cycle
       ######################
       avg_cost = 0.
       print sess.run([prediction, cost, optimizer], feed_dict={x: batch_x, y: target_y})   
       print "Target: ", temp_y, "   prediction:", prediction.eval({x: batch_x, y: target_y})
       print "cost", cost.eval({x: batch_x, y: target_y})

       #####################
       # Display logs per epoch step
       ######################

       #if epoch % batch_size == 0:
       #     print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
       #     print "\nfrom key", sess.run(key)   # training data filename and line number
       #     board=tf.zeros([n_vertices, n_vertices])
       #print "\n\nEpoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
       
       ######################
       # Clean up for next cycle
       ######################
       del target_y[0]
       if epoch % batch_size == 0:
             board=tf.zeros([n_vertices, n_vertices])


    print "Optimization Finished!"

    coord.request_stop()
    coord.join(threads)

