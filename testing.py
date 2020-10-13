from NeuralNetwork import NeuralNetwork, adam_descent_step, simple_scheme
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(666)
iterations = 500
I = 1000
y0 = np.random.uniform(-2, 2, I)
K = 15
h = 0.1
d = 2
tau = 0.1


F = lambda y: 1/2*y**2
c = F(y0)
c = c.reshape((I, 1))
network = NeuralNetwork(K, tau, h, y0, d, c, I)

network.train_adams_descent(iterations)
network.plot_cost()

data = np.random.uniform(-2, 2, I)
network.evaluate_data(data)
plt.figure()
plt.scatter(data, F(data))
plt.scatter(data, network.yps, marker='+')
plt.grid()



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

