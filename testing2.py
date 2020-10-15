import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NeuralNetwork import NeuralNetwork, adam_descent_step, simple_scheme
import random
import timeit

plt.style.use('seaborn')

#  Investigate systematically what are optimal choices for K, tau , d, h and any
#  other choices you need to make. Balance performance in the generalisation
#  phase with time consumption of training.

def plot(x, y, x_name, y_name, title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name, rotation=0)  # turn around J(theta)
    plt.grid(True)
    plt.show()


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
        solution = F(data)
        solution = solution.reshape((I, 1))
        cost_vec[i] = network.get_average_residual(solution)
        stop = timeit.default_timer()
        time_vector[i] = stop - start
        i = i + 1
    plot(K_vector, cost_vec, 'number of layers, K', 'Average\nresidual', 'Average residual as function of layers')
    plot(K_vector, time_vector, 'number of layers, K', 'runtime\n[sek]', 'Run time as function of layers')


def test_tau(K, tau_vector, h, y0, d, c, I):
    cost_vec =  np.zeros(len(tau_vector))
    time_vector = np.zeros(len(tau_vector))
    i = 0
    for tau in tau_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        network.train_vanilla(iterations)
        data = np.random.uniform(-2, 2, I)
        network.evaluate_data(data)
        solution = F(data)
        solution = solution.reshape((I, 1))
        cost_vec[i] = network.get_average_residual(solution)
        stop = timeit.default_timer()
        time_vector[i] = stop - start
        i = i + 1
    plot(tau_vector, cost_vec, r'$\tau$', 'Average\nresidual', r'Average residual as function of $\tau$')
    plot(tau_vector, time_vector, 'learning parameter,' + r'$\tau$','runtime\n[sek]', r'Run time as function of $\tau$')


def test_h(K, tau, h_vector, y0, d, c, I):
    cost_vec =  np.zeros(len(h_vector))
    time_vector = np.zeros(len(h_vector))
    i = 0
    for h in h_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        network.train_vanilla(iterations)
        data = np.random.uniform(-2, 2, I)
        network.evaluate_data(data)
        solution = F(data)
        solution = solution.reshape((I, 1))
        cost_vec[i] = network.get_average_residual(solution)
        stop = timeit.default_timer()
        time_vector[i] =  stop - start
        i = i + 1
    plot(h_vector, cost_vec, 'h', 'Average\nresidual', 'Average residual as function of step length')
    plot(h_vector, time_vector, 'step length, h', 'runtime\n[sek]', 'Run time as function of step length' )


def test_d(K, tau, h, y0, d_vector, c, I):
    cost_vec =  np.zeros(len(d_vector))
    time_vector = np.zeros(len(d_vector))
    i = 0
    for d in d_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        network.train_vanilla(iterations)
        data = np.random.uniform(-2, 2, I)
        network.evaluate_data(data)
        solution = F(data)
        solution = solution.reshape((I, 1))
        cost_vec[i] = network.get_average_residual(solution)
        stop = timeit.default_timer()
        time_vector[i] = stop - start
        i = i + 1
    plot(d_vector, cost_vec, 'dimensions, d', 'Average\nresidual', 'Average residual as function of dimension')
    plot(d_vector, time_vector, 'dimensions,d', 'runtime\n[sek]', 'Run time as function of dimension')


def random_test( y0, c, I, iterations):
    random_K = random.randint( 10, 17)
    random_tau = round(random.uniform( 0.01, 0.9), 2)
    random_h = round(random.uniform( 0.15, 0.5), 2)
    random_d = random.randint( 2, 4)
    nn = NeuralNetwork( random_K, random_tau, random_h, y0, random_d, c, I)
    nn.train_vanilla(iterations)
    data = np.random.uniform(-2, 2, I)
    nn.evaluate_data(data)
    solution = F(data)
    solution = solution.reshape((I, 1))
    res = nn.get_average_residual(solution)
    return np.array([random_K, random_tau, random_h, random_d, res])


def print_random_test(result, all, i):
    K = result[0, :]
    tau = result[1, :]
    h = result[2, :]
    res = result[4, :]
    d = result[3, :]
    if all:
        print('K:\n', K)
        print('τ:\n', tau)
        print('h:\n', h)
        print('d:\n', d)
        print('Average residual:\n', res)
    else:
        print('K:', K[i], '\nτ:', tau[i], '\nh:', h[i], '\nd:', d[i],'\nAverage residual:',res[i])


# Testing different values for the parameters and evaluating run time
# Fixed values
K = 12
h = 0.4
d = 3
tau = 0.1
# Varied variables
K_vec = np.linspace(3, 21, 29, dtype = int)
h_vec = np.linspace(0.01, 0.9, 10)
d_vec = np.linspace(1, 8, 8, dtype = int)
tau_vec = np.linspace(0.001, 0.9, 20)

iterations = 500
I = 500
y0 = np.random.uniform(-2, 2, I)
y0.sort()
F = lambda y: 1/2*y**2
c = F(y0)
c = c.reshape((I, 1))

test_K(K_vec, tau, h, y0, d, c, I)  # [10, 17]
# test_tau(K, tau_vec, h, y0, d, c, I)  # [0.1, 0.5]?
# test_h(K, tau, h_vec, y0, d, c, I)  # [0.2, ->]
# test_d(K, tau, h, y0, d_vec, c, I)  # [2, ->]
# Ser ut til at dimmensjoner og antall lag er de som påvirker run timen mest.

def N_random_test(N):
    result = np.zeros((5, N))
    for i in range(N):
        #print(np.shape(random_test( y0, c, I, iterations)))
        result[:,i] = random_test( y0, c, I, iterations)
    x = np.linspace(0,N+1,N)
    K = result[0, :]
    tau = result[1, :]
    h = result[2, :]
    d = result[3, :]
    res = result[4, :]
    #plt.plot(x, K, label='K')
    plt.plot(x, tau, label=r'$\tau$')
    plt.plot(x, h, label='h')
    plt.plot(x, d, label='d')
    plt.plot(x, res, label='average residual')
    plt.legend()
    plt.grid(True)
    plt.show()
    return result

#result = N_random_test(20)
#print_random_test(result, True, 0)

#print('\nBest')
#i = np.where(result[4,:] == np.amin(result[4,:]))  # best
#print_random_test(result, False, i)

#print('\nWorst')
#j = np.where(result[4,:] == np.amax(result[4,:]))  # worst
#print_random_test(result, False, j)



