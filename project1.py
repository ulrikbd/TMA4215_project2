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
        self.I = len(y0)
        self.d = d
        self.d0 = np.ndim(y0)
        self.W = np.random.randn(self.K, self.d, self.d)
        self.b = np.random.randn(self.K, self.d, 1)
        self.w = np.random.randn(self.d, 1)
        self.mu = np.random.randn(1)
        self.c = c
        self.Z = None
        self.yps = None
        self.P = None


    def embed_1D(self):
        y = np.zeros(shape=(self.d, self.I))
        y[0] = self.y0
        self.y0 = y

    def initialize_Z(self):
        self.Z = np.zeros(shape=(self.K, self.d, self.I))
        self.Z[0] = self.y0
        for k in range(1, self.K):
            self.Z[k] = self.get_Z_kp1(k - 1)

    def initialize_yps(self):
        self.yps = self.hypothesis_function(np.transpose(self.Z[-1])@self.w + np.ones((self.I, 1))*self.mu)

    def initialize_P(self):
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
        print("I:", self.I)
        print("y0:", self.y0)
        print("c:", self.c)
        print("Z:", self.Z)
        print("yps:", self.yps)

    def get_P_km1(self, k):
        return self.P[k + 1] + self.h*np.transpose(self.W[k]) @ (self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]) * self.P[k+1])


    def dJ_dWk(self, k):
        return self.h*(self.P[k+1] * self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]) @ np.transpose(self.Z[k]))

    def dJ_dbk(self, k):
        s = np.shape(self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]))
        return self.h*(self.P[k+1] * self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]) @ np.ones(s))

    def dJ_dw(self, yps):
        Z_K = self.Z[-1]
        s = np.shape(np.transpose(Z_K) @ self.w)
        return Z_K @ (((yps-self.c)) * self.hypothesis_function_derivated(np.transpoose(Z_K) @ self.w + self.mu @ np.ones(np.shape(s))))

    def dJ_dmu(self, yps):
        Z_K = self.Z[-1]
        s = np.shape(np.transpose(Z_K) @ self.w)
        return self.hypothesis_function_derivated(np.transpose(np.transpose(Z_K) @ self.w + self.mu @ np.ones(np.shape(s)))) @ (yps-self.c)

def adam_descent_step(U, dU, j, m, v):
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