import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NeuralNetwork import NeuralNetwork, adam_descent_step, simple_scheme
import timeit

plt.style.use('seaborn')

#  Investigate systematically what are optimal choices for K,  , d, h and any
#  other choices you need to make. Balance performance in the generalisation
#  phase with time consumption of training.

def test_K(K_vector, tau, h, y0, d, c, I):
    cost_vec =  np.zeros(len(K_vector))
    time_vector = np.zeros(len(K_vector))
    i = 0
    for K in K_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        network.train_adams_descent(iterations)
        data = np.random.uniform(-2, 2, I)
        network.evaluate_data(data)
        cost_vec[i] = network.cost[-1]
        stop = timeit.default_timer()
        time_vector[i] = stop - start
        i = i + 1
    plt.plot(K_vector, cost_vec)
    plt.xlabel('number of layers, K')
    plt.ylabel(r'$J(\theta)$', rotation=0) # turn around J(theta)
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(K_vector, time_vector)
    plt.xlabel('number of layers, K')
    plt.ylabel('runtime [sek]')
    plt.grid(True)
    plt.show()


def test_tau(K, tau_vector, h, y0, d, c, I):
    cost_vec =  np.zeros(len(tau_vector))
    time_vector = np.zeros(len(tau_vector))
    i = 0
    for tau in tau_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        network.train_adams_descent(iterations)
        data = np.random.uniform(-2, 2, I)
        network.evaluate_data(data)
        cost_vec[i] = network.cost[-1]
        stop = timeit.default_timer()
        time_vector[i] = stop - start
        i = i + 1
    plt.plot(tau_vector, cost_vec)
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$J(\theta)$', rotation=0) # turn around J(theta)
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(tau_vector, time_vector)
    plt.xlabel('learning parameter,' + r'$\tau$')
    plt.ylabel('runtime [sek]')
    plt.grid(True)
    plt.show()


def test_h(K, tau, h_vector, y0, d, c, I):
    cost_vec =  np.zeros(len(h_vector))
    time_vector = np.zeros(len(h_vector))
    i = 0
    for h in h_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        network.train_adams_descent(iterations)
        data = np.random.uniform(-2, 2, I)
        network.evaluate_data(data)
        cost_vec[i] = network.cost[-1]
        stop = timeit.default_timer()
        time_vector[i] =  stop - start
        i = i + 1
    plt.plot(h_vector, cost_vec)
    plt.xlabel('h')
    plt.ylabel(r'$J(\theta)$', rotation=0) # turn around J(theta)
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(h_vector, time_vector)
    plt.xlabel('step length, h')
    plt.ylabel('runtime [sek]')
    plt.grid(True)
    plt.show()


def test_d(K, tau, h, y0, d_vector, c, I):
    cost_vec =  np.zeros(len(d_vector))
    time_vector = np.zeros(len(d_vector))
    i = 0
    for d in d_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        network.train_adams_descent(iterations)
        data = np.random.uniform(-2, 2, I)
        network.evaluate_data(data)
        cost_vec[i] = network.cost[-1]
        stop = timeit.default_timer()
        time_vector[i] =  stop - start
        i = i + 1
    plt.plot(d_vector, cost_vec)
    plt.xlabel('dimensions, d')
    plt.ylabel(r'$J(\theta)$', rotation=0) # turn around J(theta)
    plt.grid(True)
    plt.show()

    plt.plot(d_vector, time_vector)
    plt.xlabel('dimensions, d')
    plt.ylabel('runtime [sek]')
    plt.grid(True)
    plt.show()


np.random.seed(666)
iterations = 500
I = 500
y0 = np.random.uniform(-2, 2, I)
K_vec = np.linspace(3, 25, 26, dtype = int)
K = 10
h_vec = np.linspace(0.001, 0.9, 20)
h = 0.4
d_vec = np.linspace(1, 6, 6, dtype = int)
d = 2
tau_vec = np.linspace(0.001, 0.9, 20)
tau = 0.1
F = lambda y: 1/2*y**2
c = F(y0)
c = c.reshape((I, 1))


test_K(K_vec, tau, h, y0, d, c, I)
test_tau(K, tau_vec, h, y0, d, c, I)
test_h(K, tau, h_vec, y0, d, c, I)
test_d(K, tau, h, y0, d_vec, c, I)
