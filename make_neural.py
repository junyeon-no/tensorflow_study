from sigmoid_draw import sigmoid
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print('A1 =' ,  A1)
Z1 = sigmoid(A1)
print('Z1 = ', Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3 , 0.6]])
B2 = np.array([[0.1, 0.2]])

print('z1 = ', Z1.shape)
print('w2 = ', W2.shape)
print('b2 = ', B2.shape)

A2 = np.dot(Z1, W2)+B2
Z2 = sigmoid(A2)
print('Z2 = ' , Z2)

#출력함수 시그마
def identity_funtion(x):
    return x
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_funtion(A3)
print(A3)