from random import seed
from random import random
from math import exp

# Initialize a network
def Initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    # Hidden layer
    hidden_layer = [
        {
            'weights': [random() for i in range(n_inputs + 1)]# 1 for bias
        } for i in range(n_hidden)
    ]
    network.append(hidden_layer)
    # Output layer
    output_layer = [
        {
            'weights': [random() for i in range(n_hidden + 1)]# 1 for bias
        } for i in range(n_outputs)
    ]
    network.append(output_layer)

    return network

# Calculate neuron activation for an input
def Activate(weights, inputs):
    activation = weights[-1]# Assume that the last element of weights array is the bias
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    
    return activation

# Transfer neuron activation
def Transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Calculate the derivative of an neuron output
def Transfer_derivative(output):
    return output * (1.0 - output)

# Forward propagate input to a network output
def Forward_propagate(network, row):
    inputs = row
    for layer in network:
        outputs = []
        for neuron in layer:
            activation = Activate(neuron['weights'], inputs)
            neuron['output'] = Transfer(activation)
            outputs.append(neuron['output'])
        inputs = outputs
    
    return inputs

# Backpropagate error and store in neurons
def Backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append( 2 * (neuron['output'] - expected[j]) )
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * Transfer_derivative(neuron['output'])

# Update network weights with error
def Update_weights(network, row, learn_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= learn_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= learn_rate * neuron['delta']

# Train a network for a fixed number of epochs
def Train_network(network, train_set, learn_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train_set:
            outputs = Forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i])**2  for i in range(len(expected))])
            Backward_propagate_error(network, expected)
            Update_weights(network, row, learn_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learn_rate, sum_error))

# Make a prediction with a network
def Predict(network, row):
    outputs = Forward_propagate(network, row)
    return outputs.index(max(outputs))

# Test forward propagation
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = Initialize_network(n_inputs, 2, n_outputs)
Train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
	print(layer)