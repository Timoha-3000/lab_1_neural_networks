import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import time as pTime

#  3.4
def sigmoid_activation(v, a=1):
    return 1 / (1 + np.exp(-a * v))


def tanh_activation(v, a=1):
    return np.tanh(v / a)


def linear_threshold_activation(v, a=0):
    return np.where(v >= a, 1, 0)


def unit_step_activation(v, a=0):
    return np.where(v >= a, 1, 0)

def plot_sigmoid_activation(v, a=1):
    activation_sigmoid = sigmoid_activation(v, a)
    activation_tanh = tanh_activation(v, a)
    activation_linear = linear_threshold_activation(v, a)
    activation_unit = unit_step_activation(v, a)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(v, activation_unit, label='Unit Step Activation Function')
    axis[0, 0].set_xlabel('Input')
    axis[0, 0].set_ylabel('Output')
    axis[0, 0].set_title("Unit Step Activation Function")
    axis[0, 0].grid(True)

    axis[0, 1].plot(v, activation_sigmoid, label='Sigmoid Activation Function')
    axis[0, 1].set_xlabel('Input')
    axis[0, 1].set_ylabel('Output')
    axis[0, 1].set_title("Sigmoid Activation Function")
    axis[0, 1].grid(True)

    axis[1, 0].plot(v, activation_tanh, label='Tanh Activation Function')
    axis[1, 0].set_xlabel('Frequency, Hz')
    axis[1, 0].set_ylabel('Error')
    axis[1, 0].set_title("Tanh Activation Function")
    axis[1, 0].grid(True)

    axis[1, 1].plot(v, activation_linear, label='Linear Threshold Activation Function')
    axis[1, 1].set_xlabel('Input')
    axis[1, 1].set_ylabel('Output')
    axis[1, 1].set_title("Linear Threshold Activation Function")
    axis[1, 1].grid(True)

    plt.show()

v = np.linspace(-10, 10, 100)  # Генерация массива входных данных от -10 до 10
plot_sigmoid_activation(v)

# 3.6

tn = np.linspace(0, 10, 100)

a = 5
activation_sigmoid = sigmoid_activation(tn, a)
activation_tanh = tanh_activation(tn, a)
activation_linear = linear_threshold_activation(tn, a)
activation_unit = unit_step_activation(tn, a)

matrix_sigmoid = np.column_stack((tn, activation_sigmoid))
print("matrix_sigmoid\n", matrix_sigmoid)

matrix_tanh = np.column_stack((tn, activation_tanh))
print("matrix_sigmoid\n", matrix_sigmoid)

matrix_linear = np.column_stack((tn, activation_linear))
print("matrix_sigmoid\n", matrix_sigmoid)

matrix_unit = np.column_stack((tn, activation_unit))
print("matrix_sigmoid\n", matrix_sigmoid)

#  3.5
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-5, 5, 100)
sigmoid_derivative_values = sigmoid_derivative(x)
plt.figure(figsize=(10, 6))
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, sigmoid_derivative_values, label='Сигмовидная производная', linestyle='-.', color='g')
plt.plot(x, sigmoid(x), label='Сигмовидная функция', color='m', linestyle='--')
plt.title('Сигмовидная функция и ее производная')
plt.grid(True)
plt.legend()
plt.show()