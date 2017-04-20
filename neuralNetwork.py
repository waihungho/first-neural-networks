import numpy
import scipy.special
import matplotlib.pyplot

# -------------------------------------------------------------------------------------------------------
# mnist_train_100.csv : Training data
# mnist_test_10.csv   : Test data
#
# The expected result : (the value of possibilities may vary slightly)
# Predict: 7
# The possibilities of 0 ~ 9:
# [[ 0.03852424]    --> the possibility of digit:0
# [ 0.01248606]     --> the possibility of digit:1
# [ 0.00421341]
# [ 0.06930358]
# [ 0.04758163]
# [ 0.02241341]
# [ 0.0031377 ]
# [ 0.81715265]     --> the maximum likelihood
# [ 0.02958812]
# [ 0.02034178]]
# -------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
# Neural Network 
# -------------------------------------------------------------------------------------------------------
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes for each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)

        self.lr = learningrate
        pass
            
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        

# -------------------------------------------------------------------------------------------------------
# Main 
# -------------------------------------------------------------------------------------------------------

# 1. Configure the neural networks
input_nodes = 784 
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# 2. Train the neural networks
training_data_file = open("mnist_train_100.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes)+0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    #matplotlib.pyplot.imshow(image_array, cmap='Greys',interpolation='None')


# 3. Make prediction by the neural networks 
test_data_file = open("mnist_test_10.csv",'r') 
test_data_list = test_data_file.readlines()
test_data_file.close()

all_values=test_data_list[0].split(',')
print("Predict:", all_values[0])

result = n.query((numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01)
print("The possibilities of digits: 0 ~ 9:")
print(result)






