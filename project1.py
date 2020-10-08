import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    """
    Neural network description
    """

    def __init__(self, K, tau, h, y0, d, c):
        """Initialize varibles"""
        self.K = K
        self.tau = tau
        self.h = h
        self.y0 = y0
        self.d = d
        self.d0 = np.ndim(y0)
        self.W = np.random.randn(self.K, self.d, self.d)
        self.b = np.random.randn(self.K, self.d, 1)
        self.w = np.random.randn(self.d, 1)
        self.mu = np.random.randn(1)
        self.c = c


    def embed_1D(self):
        y = np.zeros(shape=(len(self.y0), len(self.y0)))
        y[0] = self.y0
        self.y0 = y

    def activation_function(self, x):
        return np.tanh(x)

    def hypothesis_function(self, x):
        return 1/2 * (1 + np.tanh(x/2))

    def transformation(self, y):
        pass

    def scale_y0(self):
        a = np.min(self.y0)
        b = np.max(self.y0)
        self.y0 = (self.y0 - a) / (b - a)

    def scale_c(self):
        a = np.min(self.c)
        b = np.max(self.c)
        self.c = (self.c - a) / (b - a)

    def scale_input(self):
        self.scale_y0()
        self.scale_c()

    def printparameters(self):
        print("K:", self.K)
        print("tau:", self.tau)
        print("d0", self.d0)
        print("y0:", self.y0)
        print("c:", self.c)



y0 = np.random.uniform(-2, 2, 20)
K = 3
h = 0.5
d = 2
tau = 0.5
F = lambda y: 1/2*y**2
c = F(y0)
network = NeuralNetwork(K, tau, h, y0, d, c)
network.scale_input()
network.embed_1D()
network.printparameters()

