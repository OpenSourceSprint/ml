import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data/train.csv').values
np.random.shuffle(data)

X = data[:, 1:] / 255.0
Y = data[:, 0]

m_train = 40000
X_train, Y_train = X[:m_train].T, Y[:m_train]
X_dev, Y_dev = X[m_train:].T, Y[m_train:]

def initialize_parameters():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def forward_propagation(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
    return Z1, A1, Z2, A2

def backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = np.eye(10)[Y].T
    m = Y.shape[0]
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def compute_loss(A2, Y):
    one_hot_Y = np.eye(10)[Y].T
    m = Y.shape[0]
    loss = -np.sum(one_hot_Y * np.log(A2)) / m
    return loss

def compute_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = initialize_parameters()
    losses = []
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        loss = compute_loss(A2, Y)
        losses.append(loss)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = compute_accuracy(predictions, Y)
            print(f"Iteration {i}: Loss {loss:.4f}, Accuracy {accuracy:.4f}")
    return W1, b1, W2, b2, losses

alpha = 0.01
iterations = 100
W1, b1, W2, b2, losses = gradient_descent(X_train, Y_train, alpha, iterations)

plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def evaluate_accuracy(X, Y, W1, b1, W2, b2):
    predictions = make_predictions(X, W1, b1, W2, b2)
    accuracy = compute_accuracy(predictions, Y)
    print(f"Accuracy: {accuracy:.4f}")

evaluate_accuracy(X_dev, Y_dev, W1, b1, W2, b2)
