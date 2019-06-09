################################################################################
# https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
################################################################################

import tensorflow as tf

# First, create a TensorFlow constant
const = tf.constant(2.0, name="const")

# Create TensorFlow variables
b = tf.Variable(2.0, name='b')
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

    # We now compute the output of the graph
    # a is an operation, not a variable and therefore it can be run
    # we assign the output to a_out, the value of which we print out
    # Note, we can defined operations d and e, without explicitly running them
    a_out = sess.run(a)

    print("Variable a is {}".format(a_out))
