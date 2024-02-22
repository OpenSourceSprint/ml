import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  


def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


y = np.array([[0], [1], [1], [0]])


inputLayerNeurons = X.shape[1]
hiddenLayerNeurons = 2  
outputLayerNeurons = 1


hiddenWeights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hiddenBias = np.zeros((1, hiddenLayerNeurons), dtype=float)
outputWeights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
outputBias = np.zeros((1, outputLayerNeurons), dtype=float)
epochs = 50000
learning_rate = 0.1  

for _ in range(epochs):
    hiddenLayerActivation = np.dot(X, hiddenWeights)
    hiddenLayerActivation += hiddenBias
    hiddenLayerOutput = sigmoid(hiddenLayerActivation)

    outputLayerActivation = np.dot(hiddenLayerOutput, outputWeights)
    outputLayerActivation += outputBias
    predictedOutput = sigmoid(outputLayerActivation)

    error = y - predictedOutput
    d_predictedOutput = error * sigmoid_derivative(predictedOutput)

    error_hiddenLayer = d_predictedOutput.dot(outputWeights.T)
    d_hiddenLayer = error_hiddenLayer * sigmoid_derivative(hiddenLayerOutput)

    outputWeights += hiddenLayerOutput.T.dot(d_predictedOutput) * learning_rate
    outputBias += np.sum(d_predictedOutput, axis=0, keepdims=True) * learning_rate
    hiddenWeights += X.T.dot(d_hiddenLayer) * learning_rate
    hiddenBias += np.sum(d_hiddenLayer, axis=0, keepdims=True) * learning_rate

print("Final hidden weights:", hiddenWeights)
print("Final hidden bias:", hiddenBias)
print("Final output weights:", outputWeights)
print("Final output bias:", outputBias)

hiddenLayerActivation = np.dot(X, hiddenWeights)
hiddenLayerActivation += hiddenBias
hiddenLayerOutput = sigmoid(hiddenLayerActivation)

outputLayerActivation = np.dot(hiddenLayerOutput, outputWeights)
outputLayerActivation += outputBias
finalOutput = sigmoid(outputLayerActivation)

print("\nOutput from the neural network after training:")
for i in finalOutput:
    print(round(i[0]))
