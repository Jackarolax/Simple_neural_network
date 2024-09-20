import csv2trainingdata
import jsonnetwork
import neuralNetwork
import os

layer_sizes = [784,16,16,10]
rand_weights, rand_biases = neuralNetwork.createRandomNetwork(layer_sizes)


json_filename = "savedneuralnetwork2.json"

if not os.path.isfile(json_filename):
    savedNeuralNetwork = jsonnetwork.save_neuralnetwork_to_json(json_filename, layer_sizes, rand_weights, rand_biases)
else:
    savedNeuralNetwork = jsonnetwork.load_neuralnetwork_from_json(json_filename)


learningNetwork = neuralNetwork.SimpleNeuralNetwork(savedNeuralNetwork)

training_data = csv2trainingdata.get_trainingdata(40000)


for training_example in training_data:
    inputvector = training_example[0]
    targetvector = training_example[1]

    learningNetwork.backprop(inputvector, targetvector)

new_weights = learningNetwork.weights
new_biases = learningNetwork.biases

savedNeuralNetwork = jsonnetwork.save_neuralnetwork_to_json(json_filename, layer_sizes, new_weights, new_biases)