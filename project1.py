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
        self.Z = []


    def embed_1D(self):
        y = np.zeros(shape=(self.d, len(self.y0)))
        y[0] = self.y0
        self.y0 = y

    def initialize_Z(self):
        self.Z = np.zeros(shape=(self.K, self.d, len(self.y0[0])))
        self.Z[0] = self.y0

    def activation_function(self, x):
        return np.tanh(x)

    def activation_function_derivated(self, x):
        return 1/(np.cosh(x)**2)

    def hypothesis_function(self, x):
        return 1/2 * (1 + np.tanh(x/2))

    def hypothesis_function_derivated(self, x):
        return 1/(2 + 2*np.cosh(x))

    def transformation(self, y, k):
         return y + self.h*self.activation_function(self.W[k] @ y + self.b[k])

    def get_Z_kp1(self, k):
        return self.transformation(self.Z[k], k)

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

    def get_P_km1(self, k):
        # Y is the vector of function values from the last layer.
        return self.P[k+1] + np.outer(self.h*np.transpose(self.W[k]) , (self.activation_function_derivated(self.W[k-1] @ self.Z[k-1] + self.b[k-1]) * P_k))

    def dJ_dWk(self, k, P_kp1):
        return self.h*(P_kp1 * self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]) @ np.transpose(self.Z[k]))

    def dJ_dbk(self):
        s = np.shape(self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]))
        return self.h*(P_kp1 * self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]) @ np.ones(s))

    def dJ_dw(self, yps):
        Z_K = self.Z[-1]
        s = np.shape(np.transpose(Z_K) @ self.w)
        return Z_K @ (((yps-self.c)) * self.hypothesis_function_derivated(np.transpoose(Z_K) @ self.w + self.mu @ np.ones(np.shape(s))))

    def dJ_dmu(self, yps):
        Z_K = self.Z[-1]
        s = np.shape(np.transpose(Z_K) @ self.w)
        return self.hypothesis_function_derivated(np.transpose(np.transpose(Z_K) @ self.w + self.mu @ np.ones(np.shape(s)))) @ (yps-self.c)

    def adam_decent(self):
        P_kp1 = 0
        beta1 = 0.9
        beta2 = 0.999
        alpha = 0.01
        e = 10**(-8)
        v0 = np.zeros(2*self.K) #??
        v0 = np.zeros(2*self.K) #??
        g = np.zeros(2*self.K+2)
        g[-1] = self.dJ_dmu(Y)
        g[-2] = self.dJ_dw(Y)
        for j in range(1,K):
            g[0,j] = self.dJ_dWk(self, j, P_kp1)
            g[1,j] = self.

            m[j] = beta1*m[j-1] + (1-beta1)*gj




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
network.initialize_Z()
