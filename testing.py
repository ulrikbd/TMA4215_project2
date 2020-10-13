from NeuralNetwork import NeuralNetwork, adam_descent_step, simple_scheme
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    iterations = 500
    I = 300
    y0 = np.random.uniform(-2, 2, I)
    K = 20
    h = 0.1
    d = 2
    tau = 0.1
    c = F(y0)
    c = c.reshape((I, 1))
    network = NeuralNetwork(K, tau, h, y0, d, c, I)

    # Train the model using the vanilla gradient descent
    network.train_vanilla(iterations)
    # Plot the cost function
    network.plot_cost()
    plt.savefig("./plots/cost_func_F(y)_vanilla.pdf", bbox_inches="tight")

    data = np.random.uniform(-2, 2, I)
    x = np.linspace(-2, 2)
    network.evaluate_data(data)
    plt.figure()
    plt.plot(x, F(x), label="Exact solution")
    plt.scatter(data, network.yps, marker='.', c="r", s=7, label="Model solution")
    plt.xlabel(r'$y$')
    plt.grid()
    plt.legend()
    plt.savefig("./plots/test_solution_F(y)_vanilla.pdf", bbox_inches="tight")


    y0 = np.random.uniform(-np.pi/3, np.pi/3, I)
    K = 20
    c = G(y0)
    c = c.reshape((I, 1))
    network = NeuralNetwork(K, tau, h, y0, d, c, I)

    # Train the model
    network.train_adams_descent(iterations)
    # Plot the cost function
    network.plot_cost()
    plt.savefig("./plots/cost_func_G(y)_adams.pdf", bbox_inches="tight")

    data = np.random.uniform(-np.pi/3, np.pi/3, I)
    x = np.linspace(-np.pi/3, np.pi/3)
    network.evaluate_data(data)
    plt.figure()
    plt.plot(x, G(x), label="Exact solution")
    plt.scatter(data, network.yps, marker='.', c="r", s=7, label="Model solution")
    plt.xlabel(r'$y$')
    plt.grid()
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
    plt.savefig("./plots/cost_func_H(y)_vanilla.pdf", bbox_inches="tight")

    data1 = np.random.uniform(-2, 2, I)
    data2 = np.random.uniform(-2, 2, I)
    data = np.array([data1, data2])
    nn.evaluate_data(data)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data1, data2, nn.yps)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data1, data2, H([data1, data2]))




# 2D testing
I = 300
iterations = 500
y1 = np.random.uniform(-2, 2, I)
y2 = np.random.uniform(-2, 2, I)
y0 = np.array([y1, y2])
d = 4
K = 10
h = 0.1
tau = 0.1


def G(y):
    return 1 / 2 * (y[0]**2 + y[1]**2)


c = G(y0)

c = c.reshape((I, 1))
nn = NeuralNetwork(K, tau, h, y0, d, c, I)
nn.train_vanilla(iterations)
nn.plot_cost()


data1 = np.random.uniform(-2, 2, I)
data2 = np.random.uniform(-2, 2, I)
data = np.array([data1, data2])
nn.evaluate_data(data)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data1, data2, nn.yps)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data1, data2, G([data1, data2]))

def main():
    test_with_known_functions()


if __name__ == "__main__":
    main()