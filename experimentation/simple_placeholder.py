################################################################################
# https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
################################################################################

import tensorflow as tf
import numpy as np

# First, create a TensorFlow constant
const = tf.constant(2.0, name="const")

# Create TensorFlow placeholder variable
# We don't know what the value of the array p would be during the declaration
# of the TensorFlow problem (ie. before the 'with tf.Session()') stage.
# In this case, TensorFlow requires us to declare the basic structure of the
# data by using the tf.placeholder variable declaration.
# Explanation of the arguments:
#   1. The data type stored in the tensor is tf.float32
#   2. The shape of the data is an (? x 1) array
#   3. The name of the operation is p
b = tf.placeholder(tf.float32, [None, 1], name='b')

# Create TensorFlow variable
c = tf.Variable(1.0, name='c')

# The first element in a constant or variable declaration is its initial value
# The second element is an optional name string which can be used to label
#   the constant / variable for visualizations

# Now, create some operations
d = tf.add(b, c, name='d') # d = b + c
e = tf.add(c, const, name='e') # e = c + const = c + 2
a = tf.multiply(d, e, name='a') # a = d * e = (b + c) * (c + 2)

# setup the variable initialisation
init_op = tf.global_variables_initializer()

# start the TensorFlow session, which is an object where all operations are run
with tf.Session() as sess:
    # initialize the variables
    sess.run(init_op)

    # We now compute the output of the graph.
    # The 'feed_dict' argument specifies the value of the variable b,
    # which is a one-dimensional range from 0 to 10.
    # The input to be supplied is a Python dictionary
    # (ex. {b: np.arange(0, 10)[:, np.newaxis]}) with each key being
    # the name of the placeholder that we are filling.
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})

    # TensorFlow adapts naturally from a scalar output (i.e. a singular output
    # when a=9.0) to a tensor (i.e. an array/matrix
    print("Variable a is {}".format(a_out))
