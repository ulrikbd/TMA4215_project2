import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    """
    Neural network description
    """

    def __init__(self, K, tau, h, y0, d, c):
        """Initialize varibles"""
        self.K = K # number of layers
        self.tau = tau # learning parameter
        self.h = h # step length
        self.y0 = y0 # input data
        self.I = len(y0) # number of data points
        self.d = d # dimension of the hidden layers
        self.d0 = np.ndim(y0) # dimension of input data
        self.W = np.random.randn(self.K, self.d, self.d) # weights
        self.b = np.random.randn(self.K, self.d, 1) # bias
        self.w = np.random.randn(self.d, 1) # parameter for the last hidden layer
        self.mu = np.random.randn(1) # parameter for the last hidden layer
        self.c = c # vector of given data points
        self.Z = None # intermediate values
        self.yps = None # vector of function values
        self.P = None # vector of back propagation

    def embed_1D(self): # embed input data to dimension d
        y = np.zeros(shape=(self.d, self.I))
        y[0] = self.y0
        self.y0 = y

    def initialize_Z(self): # making Z
        self.Z = np.zeros(shape=(self.K, self.d, self.I))
        self.Z[0] = self.y0
        for k in range(1, self.K):
            self.Z[k] = self.get_Z_kp1(k - 1)

    def initialize_yps(self): # making Y
        self.yps = self.hypothesis_function(np.transpose(self.Z[-1]) @ self.w + np.ones((self.I, 1))*self.mu)

    def initialize_P(self): # making P
        self.P = np.zeros(shape=(self.K, self.d, self.I))
        self.P[-1] = self.w @ np.transpose(np.multiply((self.yps - self.c), self.hypothesis_function_derivated(np.transpose(self.Z[-1]) @ self.w + self.mu*np.ones((self.I, 1)))))
        for k in range(K - 2, -1, -1):
            self.P[k] = self.get_P_km1(k)

    def activation_function(self, x):
        return np.tanh(x)

    def activation_function_derivated(self, x):
        return 1/(np.cosh(x)**2)

    def objective_function(self):
        return 1/2 * np.linalg.norm(self.yps - self.c)**2

    def hypothesis_function(self, x):
        return 1/2 * (1 + np.tanh(x/2))

    def hypothesis_function_derivated(self, x):
        return 1/(2 + 2*np.cosh(x))

    def transformation(self, y, k):
        return y + self.h*self.activation_function(self.W[k] @ y + self.b[k])

    def get_Z_kp1(self, k): # get Z_(k+1)
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
        print("I:", self.I)
        print("y0:", self.y0)
        print("c:", self.c)
        print("Z:", self.Z)
        print("yps:", self.yps)

    def get_P_km1(self, k): # get P_(k-1)
        return self.P[k + 1] + self.h*np.transpose(self.W[k]) @ (self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]) * self.P[k+1])

    def dJ_dWk(self, k): # get dJ/dW_k which is element of theta
        return self.h*(self.P[k+1] * self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]) @ np.transpose(self.Z[k]))

    def dJ_dbk(self, k): # get dJ/db_k which is element of theta
        return self.h*(self.P[k+1] * self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k])) @ np.ones((I, 1))

    def dJ_dw(self, yps): # get dJ/dw which is element of theta
        Z_K = self.Z[-1]
        return Z_K @ ((yps-self.c) * self.hypothesis_function_derivated(np.transpose(Z_K) @ self.w + self.mu * np.ones((I, 1))))

    def dJ_dmu(self, yps): # get dJ/dmu which is element of theta
        Z_K = self.Z[-1]
        return self.hypothesis_function_derivated(np.transpose(np.transpose(Z_K) @ self.w + self.mu * np.ones((I, 1))) @ (yps-self.c))

    def train(self, method):
        for k in range(K-1):
            self.W[k] = method(self.W[k], self.dJ_dWk(k), self.tau)
            self.b[k] = method(self.b[k], self.dJ_dbk(k), self.tau)
            self.w = method(self.w, self.dJ_dw(self.yps), self.tau)
            self.mu = method(self.mu, self.dJ_dmu(self.yps), self.tau)

    #def test(self):
    #    y = np.zeros(I)
    #    y[0] = y0
    #    y[1] = self.transformation(y0, 1)
    #    for k in range(2, K):
    #        y[k] = self.transformation(y, k)
    #    return y


def adam_descent_step(U, dU, j, m, v): # One step of the adam gradient decent for one parameter
    beta_1 = 0.9
    beta_2 = 0.999
    alpha = 0.01
    eps = 10**(-8)
    g = dU
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * (g * g)
    m_hat = m/(1 - beta_1**j)
    v_hat = v/(1 - beta_2**j)
    U = U - alpha*m_hat/(np.sqrt(v_hat) + eps)
    return U, m, v


def simple_scheme(U, dU, tau): # One step of simple scheme to optimize weights and bias, for one parameter
    return U- tau * dU


I = 20
y0 = np.random.uniform(-2, 2, I)
K = 3
h = 0.5
d = 2
tau = 0.5
F = lambda y: 1/2*y**2
c = F(y0)
c = c.reshape((I, 1))
network = NeuralNetwork(K, tau, h, y0, d, c)
network.scale_input()
network.embed_1D()
network.initialize_Z()
network.initialize_yps()
network.initialize_P()
print(network.P)
network.train(simple_scheme)

























