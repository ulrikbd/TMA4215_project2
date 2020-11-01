from NeuralNetwork import NeuralNetwork, adam_descent_step, simple_scheme
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn')

np.random.seed(666)

def test_with_known_functions():
    """
    Task 1a)
    Test the model by using the suggested functions"""

    def F(y):
        """
        d0:1
        d: 2
        domain: [-2, 2]"""
        return 1/2 * y**2

    def G(y):
        """
        d0:1
        d :2
        domain: [-pi/3, pi/3]"""
        return 1 - np.cos(y)

    def H(y):
        """
        d0:2
        d :4
        domain: [-2, 2] x [-2, 2]"""
        return 1/2*(y[0]**2 + y[1]**2)

    def S(y):
        """
        d0:2
        d :4
        domain: Exclude origin"""
        return -(1 / (np.sqrt(y[0]**2 + y[1]**2)))

    iterations = 1000
    I = 1000
    y0 = np.random.uniform(-2, 2, I)
    K = 15
    h = 0.1
    d = 2
    tau = 0.08
    c = F(y0)
    c = c.reshape((I, 1))
    network = NeuralNetwork(K, tau, h, y0, d, c, I, scale=True)
    # Train the model using the vanilla gradient descent
    network.train_vanilla(iterations)
    # Plot the cost function
    network.plot_cost()
    plt.grid(True)
    plt.savefig("./plots/cost_func_F(y)_vanilla.pdf", bbox_inches="tight")
    plt.show()

    data = np.random.uniform(-2, 2, I)
    x = np.linspace(-2, 2)
    network.evaluate_data(data)
    plt.figure()
    plt.plot(x, F(x), label="Exact solution")
    plt.scatter(data, network.yps, marker='.', c="r", s=7, label="Model solution")
    plt.xlabel(r'$y$')
    plt.grid(True)
    plt.legend()
    plt.savefig("./plots/test_solution_F(y)_vanilla.pdf", bbox_inches="tight")


    y0 = np.random.uniform(-np.pi/3, np.pi/3, I)
    K = 15
    c = G(y0)
    c = c.reshape((I, 1))
    network = NeuralNetwork(K, tau, h, y0, d, c, I)

    # Train the model
    network.train_adams_descent(iterations)
    # Plot the cost function
    network.plot_cost()
    plt.grid(True)
    plt.savefig("./plots/cost_func_G(y)_adams.pdf", bbox_inches="tight")

    data = np.random.uniform(-np.pi/3, np.pi/3, I)
    x = np.linspace(-np.pi/3, np.pi/3)
    network.evaluate_data(data)
    plt.figure()
    plt.plot(x, G(x), label="Exact solution")
    plt.scatter(data, network.yps, marker='.', c="r", s=7, label="Model solution")
    plt.xlabel(r'$y$')
    plt.grid(True)
    plt.legend()
    plt.savefig("./plots/test_solution_G(y)_vanilla.pdf", bbox_inches="tight")

    y1 = np.random.uniform(-2, 2, I)
    y2 = np.random.uniform(-2, 2, I)
    y0 = np.array([y1, y2])
    d = 4
    c = H(y0)
    c = c.reshape((I, 1))

    nn = NeuralNetwork(K, tau, h, y0, d, c, I)
    # Train the model
    nn.train_vanilla(iterations)
    # Plot the cost function
    nn.plot_cost()
    plt.grid(True)
    plt.savefig("./plots/cost_func_H(y)_vanilla.pdf", bbox_inches="tight")

    data1 = np.random.uniform(-2, 2, I)
    data2 = np.random.uniform(-2, 2, I)
    data = np.array([data1, data2])
    nn.evaluate_data(data)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(data1, data2, H([data1, data2]), alpha=0.4)
    ax.scatter3D(data1, data2, nn.yps, c='r', label='Model solution')
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')
    ax.legend()
    plt.savefig("./plots/test_solution_H(y)_vanilla.pdf", bbox_inches="tight")


    y1 = np.random.uniform(-10, 10, I)
    y2 = np.random.uniform(-10, 10, I)
    y0 = np.array([y1, y2])
    K = 20
    h = 0.08
    tau = 0.08
    d = 4
    c = S(y0)
    c = c.reshape((I, 1))

    nn = NeuralNetwork(K, tau, h, y0, d, c, I)
    # Train the model
    nn.train_vanilla(iterations)
    # Plot the cost function
    nn.plot_cost()
    plt.grid(True)
    plt.savefig("./plots/cost_func_S(y)_vanilla.pdf", bbox_inches="tight")

    data1 = np.random.uniform(-1, 1, I)
    data2 = np.random.uniform(-1, 1, I)
    data = np.array([data1, data2])
    nn.evaluate_data(data)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(data1, data2, S([data1, data2]), alpha=0.4)
    ax.scatter3D(data1, data2, nn.yps, c='r', label='Model solution')
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')
    ax.legend()
    plt.savefig("./plots/test_solution_S(y)_vanilla.pdf", bbox_inches="tight")


def test_method():
    """Generic test which can me messed around with"""
    def F(y):
        """
        d0:1
        d: 2
        domain: [-2, 2]"""
        return np.sin(8*y)
    iterations = 2000
    I = 4000
    y0 = np.random.uniform(-2, 2, I)
    K = 40
    h = 0.1
    d = 2
    tau = 0.08
    chunk_size = 100
    c = F(y0)
    c = c.reshape((I, 1))
    network = NeuralNetwork(K, tau, h, y0, d, c, I)
    # Train the model
    network.train_stochastic_gradient_descent(iterations, chunk_size)
    # Plot the cost function
    network.plot_cost()
    plt.grid(True)
    network.scale_up_solution()

    x = np.linspace(-2, 2)
    plt.figure()
    plt.scatter(y0, F(y0), label="Exact solution")
    plt.scatter(y0, network.yps, marker='.', c="r", s=7, label="Model solution")
    plt.xlabel(r'$y$')
    plt.grid(True)
    plt.show()


def main():
    """Generic test which can me messed around with"""
    # test_method()
    """Task 1a)
    Test the model by using the suggested functions"""
    test_with_known_functions()



if __name__ == "__main__":
    main()