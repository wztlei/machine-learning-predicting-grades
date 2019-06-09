################################################################################
# https://adventuresinmachinelearning.com/neural-networks-tutorial/
################################################################################


import matplotlib.pylab as plt
import numpy as np


################################################################################
NUM_GRAPHS = 3
graph_row = 1
graph_col = 1
plt.figure(figsize=(7, 8))
plt.subplots_adjust(hspace=1)

# A neuron is simulated in a neural network by an activation function, ie. when
# the input is greater than a certain value, the output should change state.
# A common activation function that is used is the sigmoid function:
#              1
# f(x) = ------------
#        1 + exp(-x))

# Declare a range for x and the sigmoid function f
x = np.arange(-10, 10, 0.1)
f = 1 / (1 + np.exp(-x))

# Plot the graph
plt.subplot(NUM_GRAPHS, graph_col, graph_row)
plt.title('Figure 1. The Sigmoid Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, f)
graph_row += 1

# Neural networks consist of connected layers of nodes, which all take multiple
# weighted inputs, applies the activation function to the sum of these inputs,
# and generates an output.
# In the equation below, xi is the input, wi is the weight, and b is the bias.
# That is: Input = x1*w1 + x2w2 + ... + wn*xn + b

# Consider a simple node with only one input and one output:
# If we change the weight from w1 = 0.5, to w2 = 1.0, to w3 = 2.0, then we
# see that changing the slope changes the slope of the sigmoid function.
# This is useful for modeling different strengths of relationships
# between the input and output variables.

# Initialize different weights and their corresponding labels
w1 = 0.5
w2 = 1.0
w3 = 2.0
l1 = 'w = 0.5'
l2 = 'w = 1.0'
l3 = 'w = 2.0'

# Plot the different sigmoid functions with different weights
plt.subplot(NUM_GRAPHS, graph_col, graph_row)
plt.title('Figure 2. Effect of adjusting the weights')
plt.xlabel('x')
plt.ylabel('h_w(x)')
plt.legend(loc=2)

for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
    f = 1 / (1 + np.exp(-x * w))
    plt.plot(x, f, label=l)

graph_row += 1

# All the graphs simulate nodes that might 'turn on' at the same x-value
# (x = 0), so we add a bias to make the node simulate a generic if function,
# ie. if (x > z) then 1 else 0. Changing the bias affects the z-value at
# which the graph 'turns on'.

# Initialize different biases and their corresponding labels
w = 5.0
b1 = -8.0
b2 = 0.0
b3 = 8.0
l1 = 'b = -8.0'
l2 = 'b = 0.0'
l3 = 'b = 8.0'

# Plot the different sigmoid functions with different biases
plt.subplot(NUM_GRAPHS, graph_col, graph_row)
plt.title('Figure 3. Effect of adjusting the bias')
plt.xlabel('x')
plt.ylabel('h_wb(x)')
plt.legend(loc=2)

for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
    f = 1 / (1 + np.exp(-(x * w + b)))
    plt.plot(x, f, label=2)

graph_row += 1

plt.show()





