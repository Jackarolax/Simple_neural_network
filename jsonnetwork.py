import json
import numpy as np

def save_neuralnetwork_to_json(filename, layer_sizes, weights, biases):
    """will save the neural network and will return it"""
    savedNeuralNetwork = {
    "layer_sizes" : layer_sizes,
    "weights" : [w.tolist() for w in weights],
    "biases" : [b.tolist() for b in biases]
}
    with open(filename, 'w') as f:
        json.dump(savedNeuralNetwork, f, indent=4)

    return(savedNeuralNetwork)

def load_neuralnetwork_from_json(file_name):
    with open(file_name, 'r') as f:
        json_neuralNetwork = json.load(f)

    layer_sizes = json_neuralNetwork["layer_sizes"]
    weights = [np.array(w) for w in json_neuralNetwork['weights']]
    biases = [np.array(b) for b in json_neuralNetwork['biases']]

    savedNeuralNetwork = {
        "layer_sizes" : layer_sizes,
        "weights" : weights,
        "biases" : biases
    }
    return(savedNeuralNetwork)
