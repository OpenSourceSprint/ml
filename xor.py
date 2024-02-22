import numpy as np

# XOR Gate #

def sig(x):
    return 1 / (1 + np.exp(-x)) 

def sigDeriv(x):
    return x * (1 - x)


input = np.array([[0,0],[0,1],[1,0],[1,1]])
target = np.array([[0],[1],[1],[0]])

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Init random weights
hiddenWeights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hiddenBias = np.zeros((1, 2), dtype = float)
outputWeights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
outputBias = np.zeros((1, 1), dtype = float)

epochs = 50000
lRate = 0.1 


for _ in range(epochs):
    
    # forward prop 
    hiddenLayerActivation = np.dot(input, hiddenWeights)
    hiddenLayerActivation += hiddenBias
    hiddenLayerOutput = sig(hiddenLayerActivation)

    outputLayerActivation = np.dot(hiddenLayerOutput, outputWeights)
    outputLayerActivation += outputBias
    predictedOutput = sig(outputLayerActivation)

    # back prop
    error = target - predictedOutput
    dPredictedOutput = error * sigDeriv(predictedOutput)

    errorHiddenLayer = dPredictedOutput.dot(outputWeights.T)
    dHiddenLayer = errorHiddenLayer * sigDeriv(hiddenLayerOutput)
    
    

    # updating weights, bias
    outputWeights += hiddenLayerOutput.T.dot(dPredictedOutput) * lRate
    outputBias += np.sum(dPredictedOutput, axis = 0, keepdims=True) * lRate
    hiddenWeights += input.T.dot(dHiddenLayer) * lRate
    hiddenBias += np.sum(dHiddenLayer, axis = 0, keepdims=True) * lRate
        

print("Final hidden weights: ", end = '')
print(*hiddenWeights)
print("Final hidden bias: ", end = '')
print(*hiddenBias)
print("Final output weights: ", end = '')
print(*outputWeights)
print("Final output bias: ", end = '')
print(*outputBias)


print("\nOutput from nn: ", end = '')
print(*predictedOutput)

