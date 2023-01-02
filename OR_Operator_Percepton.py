import numpy as np

# Perfrom OR operation on two arrays and print error and training accuracy on every epoch


class Percepton(object):
    # threshold is number of iterations to train
    # learning_rate is how fast the model learns
    # no_of_inputs is number of inputs
    def __init__(self, no_of_inputs, threshold=10, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)  # +1 for bias
    # print(np.zeros(5))

    def predict(self, inputs):
        # dot product    # self.weights[0] is bias
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1  # if summation is greater than 0 then activate
        else:
            activation = 0  # else deactivate
        return activation

    def train(self, training_inputs, labels):
        # print every epoch and error
        for _ in range(self.threshold):
            error = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error += abs(label - prediction)
                self.weights[1:] += self.learning_rate * \
                    (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
            print("error --> ", error)
            print("training accuracy --> ", 1 - error/len(labels))
            print("weights --> ", self.weights)


training_inputs = []
training_inputs.append(np.array([0, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([1, 1]))

labels = np.array([0, 1, 1, 1])

percepton = Percepton(2)
percepton.train(training_inputs, labels)


# training_inputs = []
# training_inputs.append(np.array([1, 1]))
# training_inputs.append(np.array([1, 0]))
# training_inputs.append(np.array([0, 1]))
# training_inputs.append(np.array([0, 0]))

# labels = np.array([1, 1, 1, 0])  # OR

# perceptron = Percepton(2)  # 2 inputs for OR

# perceptron.train(training_inputs, labels)

# inputs = np.array([1, 1])
# print("perceptron.predict(np.array[1,1]) --> ",
#       perceptron.predict(inputs))  # 1

# inputs = np.array([0, 1])

# print("perceptron.predict(np.array[0,1]) --> ",
#       perceptron.predict(inputs))  # 1

# print("perceptron.weights --> ", perceptron.weights)  # [ 0.   0.01  0.01]
