
import math
import torch

#class for holding all of our network parameters and hyperparameters in one object
class MyNeuralNetwork:

    #layer_sizes is an array with the size of each layer, 
    # the first layer should be the shape of the input and the final layer should be the shape of the output
    def __init__(self, layer_sizes, activation_functions):
        self.activations = activation_functions
        self.layer_sizes = layer_sizes
        assert len(self.activations) == len(self.layer_sizes) - 1, "must have an activation function for each layer"
        self.input_size = layer_sizes[0] #The first "layer" size is the size of the input
        self.weights = []
        self.bias = []
        for i in range(1, len(layer_sizes)):
            #we initialize weights inline with how torch nn.linear initializes them
            #uniformly between the - and + square root of 1 / input size
            weight_matrix = torch.rand(layer_sizes[i - 1], layer_sizes[i]) 
            sqrt_k = math.sqrt(1 / layer_sizes[i - 1])
            weight_matrix = -2 * sqrt_k * weight_matrix + sqrt_k 
            #We zero initialize bias 
            bias_vector = torch.zeros(layer_sizes[i])
            self.weights.append(weight_matrix)
            self.bias.append(bias_vector)

# Forward Pass
import torch
import torch.nn

#The shape of our input, X, is (num_features)
#The shape of our output tensor is (num_classes)
def forward_pass(nn: MyNeuralNetwork, X: torch.Tensor):
    for activation_function, weight_matrix, bias_vector in zip(nn.activations, nn.weights, nn.bias):
        #this is just matrix multiplication but we're ignoring the batch dimension
        wx = torch.matmul(weight_matrix.T, X)
        wxb = wx + bias_vector
        X = activation_function(wxb)
    return X

#Accepts an input of shape (num_features) and returns the class prediction as a tensor
def predict(nn: MyNeuralNetwork, X: torch.Tensor):
    X = forward_pass(nn, X)
    return X.argmax()

# testing to see if our forward pass works
my_nn = MyNeuralNetwork((9, 9, 6), (torch.nn.ReLU(), torch.nn.Softmax(dim=0)))
predict(my_nn, torch.from_numpy(x_test.to_numpy(dtype=np.float32))[0])
