import numpy as np

# 1. Sigmoid

def sigmoid(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid

print('Sigmoid of 5 is ' + str(sigmoid(5)))

# 2. Sigmoid derivative

def sigmoid_derivative(x):
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_derivative = sigmoid * (1 - sigmoid)
    return sigmoid_derivative

x = np.array([1, 5, 10])
sig_der = sigmoid_derivative(x)

print('Sigmoid Derivative of [1, 2, 3] : ' + str(sig_der))

# 3. Rows normalization

def normalizeRows(x):
    x_norm = np.linalg.norm(x, ord = 2, keepdims = True, axis = 1)
    x = x/x_norm
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])

print("normalizeRows(x) = " + str(normalizeRows(x)))

# 4. Softmax

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])

print("softmax(x) = " + str(softmax(x)))

# 5. L1 loss function

def L1(yhat, y):
    loss = np.sum(np.abs(yhat - y), axis = 0)
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

print("L1 = " + str(L1(yhat,y)))

# 6. L2 loss function

def L2(yhat, y):
    loss = np.dot(np.abs(yhat - y), np.abs(yhat - y))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

print("L2 = " + str(L2(yhat,y)))