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


def test_and_train_adam(network, iterations, I):
    """
    For a network, train with adam and test it on new data.
    Returns the residual of the final test on the new data points."""
    network.train_adams_descent(iterations)
    data = np.random.uniform(-2, 2, I)
    network.evaluate_data(data)
    solution = F(data)
    solution = solution.reshape((I, 1))
    return network.get_average_residual(solution)


def test_and_train_vanilla(network, iterations, I):
    """
    For a network, train with vanilla and test it on new data.
    Returns the residual of the final test on the new data points."""
    network.train_vanilla(iterations)
    data = np.random.uniform(-2, 2, I)
    network.evaluate_data(data)
    solution = F(data)
    solution = solution.reshape((I, 1))
    return network.get_average_residual(solution)


def F(y):
    return 1/2 * y**2


def G(y):
    return 1 - np.cos(y)


"""
SYSTEMATIC TEST
"""


def test_K(K_vector, tau, h, y0, d, c, I, iterations, method):
    """
    Fix all parameters but th enumer of hidden layers, K. Let K take values from a range,
    and train and test network using the different number of hidden layers. Measure the
    run time for each value of K."""
    cost_vec =  np.zeros(len(K_vector))
    time_vector = np.zeros(len(K_vector))
    i = 0
    for K in K_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        res = method(network, iterations, I)
        cost_vec[i] = res
        stop = timeit.default_timer()
        time_vector[i] = stop - start
        i = i + 1
    return cost_vec, time_vector


def test_tau(K, tau_vector, h, y0, d, c, I, iterations, method):
    """
    Fix all parameters but the learning parameter, tau. Let tau take values from a range,
    and train and test network using the different learning parameters. Measure the
    run time for each value of tau."""
    cost_vec =  np.zeros(len(tau_vector))
    time_vector = np.zeros(len(tau_vector))
    i = 0
    for tau in tau_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        res = method(network, iterations, I)
        cost_vec[i] = res
        stop = timeit.default_timer()
        time_vector[i] = stop - start
        i = i + 1
    return cost_vec, time_vector


def test_h(K, tau, h_vector, y0, d, c, I, iterations, method):
    """
    Fix all parameters but the step lenght, h. Let h take values from a range,
    and train and test network using the different step lengths. Measure the
    run time for each value of h."""
    cost_vec =  np.zeros(len(h_vector))
    time_vector = np.zeros(len(h_vector))
    i = 0
    for h in h_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        res = method(network, iterations, I)
        cost_vec[i] = res
        stop = timeit.default_timer()
        time_vector[i] = stop - start
        i = i + 1
    return cost_vec, time_vector


def test_d(K, tau, h, y0, d_vector, c, I, iterations, method):
    """
    Fix all parameters but the dimension. Let d take values from a range,
    and train and test network using the different dimensions. Measure the
    run time for each value of d."""
    cost_vec =  np.zeros(len(d_vector))
    time_vector = np.zeros(len(d_vector))
    i = 0
    for d in d_vector:
        start = timeit.default_timer()
        network = NeuralNetwork(K, tau, h, y0, d, c, I)
        res = method(network, iterations, I)
        cost_vec[i] = res
        stop = timeit.default_timer()
        time_vector[i] = stop - start
        i = i + 1
    return cost_vec, time_vector


def plot_K(K_vec, KF_cost, KF_time, KG_cost, KG_time):
    """
    Plot the result from test_K() for both F(y) and G(y). """
    plt.plot(K_vec, KF_cost, label=r'$F(y)=\frac{1}{2}y^{2}$')
    plt.plot(K_vec, KG_cost, label=r'$G(y)=1-\cos(y)$')
    plt.xlabel('number of layers, K')
    plt.ylabel('average residual')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/ares_K_adam.pdf", bbox_inches="tight")
    plt.show()

    plt.plot(K_vec, KF_time, label=r'$F(y)=\frac{1}{2}y^{2}$')
    plt.plot(K_vec, KG_time, label=r'$G(y)=1-\cos(y)$')
    plt.xlabel('number of layers, K')
    plt.ylabel('runtime [sek]')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/rtime_K_adam.pdf", bbox_inches="tight")
    plt.show()


def plot_tau(tau_vec, tauF_cost, tauF_time, tauG_cost, tauG_time):
    """
    Plot the result from test_tau() for both F(y) and G(y). """
    plt.plot(tau_vec, tauF_cost, label=r'$F(y)=\frac{1}{2}y^{2}$')
    plt.plot(tau_vec, tauG_cost, label=r'$G(y)=1-\cos(y)$')
    plt.xlabel('learning parameter,' + r'$\tau$')
    plt.ylabel('average residual')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/ares_tau_adam.pdf", bbox_inches="tight")
    plt.show()

    plt.plot(tau_vec, tauF_time, label=r'$F(y)=\frac{1}{2}y^{2}$')
    plt.plot(tau_vec, tauG_time, label=r'$G(y)=1-\cos(y)$')
    plt.xlabel('learning parameter,' + r'$\tau$')
    plt.ylabel('runtime [sek]')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/rtime_tau_adam.pdf", bbox_inches="tight")
    plt.show()


def plot_d(d_vec, dF_cost, dF_time, dG_cost, dG_time):
    """
    Plot the result from test_d() for both F(y) and G(y). """
    plt.plot(d_vec, dF_cost, label=r'$F(y)=\frac{1}{2}y^{2}$')
    plt.plot(d_vec, dG_cost, label=r'$G(y)=1-\cos(y)$')
    plt.xlabel('dimensions, d')
    plt.ylabel('Average residual')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/ares_d_adam.pdf", bbox_inches="tight")
    plt.show()

    plt.plot(d_vec, dF_time, label=r'$F(y)=\frac{1}{2}y^{2}$')
    plt.plot(d_vec, dG_time, label=r'$G(y)=1-\cos(y)$')
    plt.xlabel('dimensions,d')
    plt.ylabel('runtime [sek]')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/rtime_d_adam.pdf", bbox_inches="tight")
    plt.show()


def plot_h(h_vec, hF_cost, hF_time, hG_cost, hG_time):
    """
    Plot the result from test_h() for both F(y) and G(y). """
    plt.plot(h_vec, hF_cost, label=r'$F(y)=\frac{1}{2}y^{2}$')
    plt.plot(h_vec, hG_cost, label=r'$G(y)=1-\cos(y)$')
    plt.xlabel('step length, h')
    plt.ylabel('average residual')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/ares_h_adam.pdf", bbox_inches="tight")
    plt.show()

    plt.plot(h_vec, hF_time, label=r'$F(y)=\frac{1}{2}y^{2}$')
    plt.plot(h_vec, hG_time, label=r'$G(y)=1-\cos(y)$')
    plt.xlabel('step length, h')
    plt.ylabel('runtime [sek]')
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/rtime_h_adam.pdf", bbox_inches="tight")
    plt.show()


# Testing different values for the parameters and evaluating run time
def test_parameters(method):
    """
    Create a network and plot the results using functions above."""
    # Fixed values
    K = 12
    h = 0.4
    d = 3
    tau = 0.1
    # Varied variables
    K_vec = np.linspace(3, 21, 19, dtype=int)
    h_vec = np.linspace(0.01, 0.9, 10)
    d_vec = np.linspace(1, 8, 8, dtype=int)
    tau_vec = np.linspace(0.001, 0.9, 20)

    iterations = 500
    I = 500
    y0 = np.random.uniform(-2, 2, I)
    cF = F(y0)
    cF = cF.reshape((I, 1))
    cG = G(y0)
    cG = cG.reshape((I, 1))

    KF_cost, KF_time = test_K(K_vec, tau, h, y0, d, cF, I, iterations, method)
    KG_cost, KG_time = test_K(K_vec, tau, h, y0, d, cG, I, iterations, method)
    plot_K(K_vec, KF_cost, KF_time, KG_cost, KG_time)

    if method == test_and_train_vanilla:
        tauF_cost, tauF_time = test_tau(K, tau_vec, h, y0, d, cF, I, iterations, method)
        tauG_cost, tauG_time = test_tau(K, tau_vec, h, y0, d, cG, I, iterations, method)
        plot_tau(tau_vec, tauF_cost, tauF_time, tauG_cost, tauG_time)

    hF_cost, hF_time = test_h(K, tau, h_vec, y0, d, cF, I, iterations, method)
    hG_cost, hG_time = test_h(K, tau, h_vec, y0, d, cG, I, iterations, method)
    plot_h(h_vec, hF_cost, hF_time, hG_cost, hG_time)

    dF_cost, dF_time = test_d(K, tau, h, y0, d_vec, cF, I, iterations, method)
    dG_cost, dG_time = test_d(K, tau, h, y0, d_vec, cG, I, iterations, method)
    plot_d(d_vec, dF_cost, dF_time, dG_cost, dG_time)

#test_parameters(test_and_train_vanilla)
test_parameters(test_and_train_adam)


"""
RANDOM TEST 
"""


def random_test( y0, c, I, iterations, method):
    """
    Generate random values for the parameters and create a neural network.
    test and train the network and calculate the belonging average residual. """
    random_K = random.randint(10, 17)
    random_tau = round(random.uniform( 0.01, 0.9), 2)
    random_h = round(random.uniform( 0.15, 0.5), 2)
    random_d = random.randint(2, 4)
    nn = NeuralNetwork(random_K, random_tau, random_h, y0, random_d, c, I)
    res = method(nn, iterations, I)
    return np.array([random_K, random_tau, random_h, random_d, res])


def fixed_test(K, tau, h, d, method, n):
    """
    Create a neural network of given parameter values.
    test and train the network and calculate the belonging average residual.
    Use to test a good result from the random tests."""
    iterations = 500
    I = 500
    y0 = np.random.uniform(-2, 2, I)
    c = F(y0)
    c = c.reshape((I, 1))
    res_arr = np.zeros(n)
    for i in range(n):
        nn = NeuralNetwork(K, tau, h, y0, d, c, I)
        res_arr[i] = method(nn, iterations, I)
    return res_arr


def n_random_test(N, method):
    """
    Do n random test using random_test() and store the trials in the array result."""
    iterations = 500
    I = 500
    y0 = np.random.uniform(-2, 2, I)
    y0.sort() # ta vekk?
    c = F(y0)
    c = c.reshape((I, 1))
    result = np.zeros((5, N))
    for i in range(N):
        result[:,i] = random_test(y0, c, I, iterations, method)
    x = np.linspace(0,N+1,N)
    K = result[0, :]
    tau = result[1, :]
    h = result[2, :]
    d = result[3, :]
    res = result[4, :]
    return result


def print_random_test(result, all, i):
    """
    Print either one random_test or all random tests in a array of random tests."""
    K = result[0, :]
    tau = result[1, :]
    h = result[2, :]
    res = result[4, :]
    d = result[3, :]
    if all == True:
        print('K:\n', K)
        print('τ:\n', tau)
        print('h:\n', h)
        print('d:\n', d)
        print('Average residual:\n', res)
    else:
        print('K:', K[i], '\nτ:', tau[i], '\nh:', h[i], '\nd:', d[i],'\nAverage residual:',res[i])


def test_random_parameters(method, N):
    """
    N random tests for the function F. The best test (lowest average residual) and the worst test
    (highest average residual) is printed with belonging parameter values."""
    result = n_random_test(N, method)
    print_random_test(result, True, 0)

    print('\nBest')
    i = np.where(result[4,:] == np.amin(result[4,:]))  # best
    print_random_test(result, False, i)

    print('\nWorst')
    j = np.where(result[4,:] == np.amax(result[4,:]))  # worst
    print_random_test(result, False, j)



#print('Adam')
#test_random_parameters(test_and_train_adam, 200)

#res = fixed_test(13, 0.45, 0.27, 4, test_and_train_adam, 20)
#print(np.mean(res))

