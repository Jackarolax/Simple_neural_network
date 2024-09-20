import numpy as np
import time
import csv
import json
import os
np.set_printoptions(precision=3, threshold=100, edgeitems=3, linewidth=100, suppress=True)


def vectortonum(vector):
    num = 11
    for i, value in enumerate(vector):
        if value[0] == 1:
            num = i
    if num == 11:
        raise Exception("1 not founnd in vector")
    return(num)


def avg_value(vector):
    values = []
    for value in vector:
        values.append(value[0])
    return(sum(values)/len(values))

def createRandomNetwork(layer_sizes):

    weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
    biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    return(weights, biases)


class SimpleNeuralNetwork:
    def __init__(self, savedNeuralNetwork, learning_rate = 0.5): 
        """ex. layer_sizes = [6,4,3]"""
        self.layer_sizes = savedNeuralNetwork["layer_sizes"]
        self.weights = savedNeuralNetwork["weights"]
        self.biases = savedNeuralNetwork["biases"]

        self.learning_rate = learning_rate
        self.activations = []
        self.z_values = []

    def feedforward(self, inputvector):
        
        a = inputvector
        """Return the output of the network if 'a' is input."""
        self.activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
            self.activations.append(a)
            self.z_values.append(z)
        #print(self.z_values)
        #print(self.activations)
        
        return a

    def sigmoid(self, z):
        """The sigmoid function."""
        return(1.0 / (1.0 + np.exp(-z)))
    
    def sigmoid_derivative(self, z):
        """z is the vector of W * a(-1) + b"""
        sig = self.sigmoid(z)
        return(sig * (1 - sig))
        

    def backprop(self, inputvector, targetvector):
        start = time.time()
        self.feedforward(inputvector)
        print("before")
        self.calculateerror(inputvector, targetvector)

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta = (self.activations[-1] - targetvector) * self.sigmoid_derivative(self.z_values[-1])

        #first backprop
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, self.activations[-2].T)


        #rest of the backprop
        for step in range(2, len(self.activations)):
            z = self.z_values[-step]
            delta_sigmoid = self.sigmoid_derivative(z)

            delta = np.dot(self.weights[-step + 1].T, delta) * delta_sigmoid

            nabla_b[-step] = delta
            nabla_w[-step] = np.dot(delta, np.array(self.activations[-step - 1]).T)
        
        #update the weights and biases

        self.weights = [w - (self.learning_rate * nw) for w, nw in zip(self.weights,nabla_w)]
        self.biases = [b - (self.learning_rate * nb) for b, nb in zip(self.biases,nabla_b)]

        end = time.time()
        duration = end - start
        print(f"after {duration} sec")
        self.calculateerror(inputvector, targetvector)

    def calculateerror(self, inputvector, targetvector ): # input : list, target: int
        if not len(inputvector) == self.layer_sizes[0]:
            raise Exception("Input does't match layer size")
        
        outputvector = self.feedforward(inputvector)
        print(outputvector)

        costvector = (outputvector - targetvector)**2
        print(vectortonum(targetvector))
        print(avg_value(costvector))





