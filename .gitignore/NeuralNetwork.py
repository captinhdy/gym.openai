from random import seed
from random import random
from math import exp

class NeuralNetwork:

    def _init_(self):
        self.network
        seed(1)

    def initialize_network(self, n_inputs, n_hidden,  n_outputs):

        self.network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        #hidden_layer2 = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_hidden2)]
        #self.network.append(hidden_layer2)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)

        return self.network

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            if isinstance(inputs[i], float):
                activation += weights[i] * inputs[i]
            else:
                for j in range(len(inputs[i])):
                    activation += weights[i] * inputs[i][j]
        return activation

    def transfer(self, activation):
        e = exp(-activation)
        return 1.0 / (1.0 + e)

    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])

            inputs = new_inputs

        return inputs;

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def backward_propigate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i-1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    if isinstance(inputs[i], float):
                        neuron['weights'][j] += l_rate * neuron['delta']* inputs[j]
                    else:
                        for k in range(len(inputs[j])):
                            neuron['weights'][j] += l_rate * neuron['delta']* inputs[j][k]
                neuron['weights'][-1] += l_rate * neuron['delta']

    def train_network(self, train, l_rate, n_epoch, n_outputs, value):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = value
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propigate_error(expected)
                self.update_weights(row, l_rate)
            #print('>epoch=%d, lrate=%.3f, error=%.3f' %(epoch, l_rate, sum_error))

    def predict(self, row):
        outputs = self.forward_propagate(row)
        #for i in range(len(outputs)):
            #print('Prediction for %d %.3f%%' %(i, outputs[i]))
        
        prediction = outputs.index(max(outputs))

        #print('guess is %d' % prediction)
        return prediction

    def back_propagation(self, train, test, l_rate, n_epoch, n_hidden):
        n_inputs = len(train[0]) - 1
        n_outputs = 3#len(set([row[-1] for row in train]))
        #self.network = self.initialize_network(n_inputs, n_hidden, n_outputs)
        self.train_network(train, l_rate, n_epoch, n_outputs)
        predictions = list()
        for row in test:
            prediction = self.predict(row)
            predictions.append(prediction)
            
        return(predictions)

    # Find the min and max values for each column
    def dataset_minmax(self, dataset):
        minmax = list()
        stats = [[min(column), max(column)] for column in zip(*dataset)]
        return stats
 
    # Rescale dataset columns to the range 0-1
    def normalize_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
    # Split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split
 
    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0