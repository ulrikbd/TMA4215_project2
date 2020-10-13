import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    """
    The Neural Network class with ResNet architecture. Contains the weights,
    and memberfunctions to train based on input data, evaluate the solution
    with given data, and plot the results.
    """

    def __init__(self, K, tau, h, y0, d, c, I):
        """Initialize vaiables from the parameters"""
        self.K = K  # number of layers
        self.tau = tau  # learning parameter
        self.h = h  # step length
        self.input = y0  # storing the initial input data
        self.y0 = y0  # input data
        self.I = I  # number of data points
        self.d = d  # dimension of the hidden layers
        self.d0 = np.ndim(y0)  # dimension of input data
        self.W = np.random.randn(self.K, self.d, self.d)  # weights
        self.b = np.random.randn(self.K, self.d, 1)  # bias
        # parameter for the last hidden layer
        self.w = np.random.randn(self.d, 1)
        self.mu = np.random.randn(1)  # parameter for the last hidden layer
        self.c = c  # vector of given data points
        self.alpha = 0  # parameter used for minmax scaling
        self.beta = 1  # parameter used for minmax scaling

        self.scale_input()
        self.embed()

    def embed(self):
        """Embed starting values into a higher dimension"""
        y = np.zeros(shape=(self.d, self.I))
        if self.d0 == 1:
            y[0] = self.y0
        else:
            for i in range(self.d0):
                y[i] = self.y0[i]
        self.y0 = y

    def initialize_Z(self):
        """Initialize the KxdxI matrix where all the data in the
        hidden layers are stored"""
        self.Z = np.zeros(shape=(self.K, self.d, self.I))
        self.Z[0] = self.y0
        for k in range(1, self.K):
            self.Z[k] = self.get_Z_kp1(k - 1)

    def initialize_yps(self):
        """Creating the Ypsilon-vector, which is the current solution"""
        self.yps = self.hypothesis_function(np.transpose(
            self.Z[-1]) @ self.w + np.ones((self.I, 1)) * self.mu)

    def initialize_P(self):
        """Creating the KxdxI P-matrix by back propagation. These
        values are use to calculate the gradient"""
        self.P = np.zeros(shape=(self.K, self.d, self.I))
        self.P[-1] = self.w @ np.transpose(np.multiply((self.yps - self.c), self.hypothesis_function_derivated(
            np.transpose(self.Z[-1]) @ self.w + self.mu * np.ones((self.I, 1)))))
        for k in range(self.K - 2, -1, -1):
            self.P[k] = self.get_P_km1(k)

    def activation_function(self, x):
        """Non-linear scalar activation function"""
        return np.tanh(x)

    def activation_function_derivated(self, x):
        """Derivative of the activation function"""
        return 1 - np.tanh(x)**2

    def objective_function(self):
        return 1 / 2 * np.linalg.norm(self.yps - self.c)**2

    def hypothesis_function(self, x):
        """Hypothesis function, could be omitted"""
        return 1 / 2 * (1 + np.tanh(x / 2))

    def hypothesis_function_derivated(self, x):
        """Derivative of the hypothesis function"""
        return 1/4*(1-np.tanh(x/2)**2)

    def transformation(self, y, k):
        """Function which maps from one layer to the next in the
        neural networks"""
        return y + self.h * self.activation_function(self.W[k] @ y + self.b[k])

    def get_Z_kp1(self, k):
        """Supplement function which returns the value for Z at the
        next layerss"""
        return self.transformation(self.Z[k], k)

    def get_scaling_factors(self):
        """Find the values for maxmin scaling both in the input, and in
        the given data"""
        self.y0_a = np.min(self.y0)
        self.y0_b = np.max(self.y0)
        self.c_a = np.min(self.c)
        self.c_b = np.max(self.c)

    def scale_y0(self):
        """Scales the input values with maxmin scaling"""
        self.y0 = 1 / (self.y0_b - self.y0_a) * (
            (self.y0_b - self.y0) * self.alpha + (self.y0 - self.y0_a) * self.beta)

    def scale_c(self):
        """Scales the given datapoints with maxmin scaling"""
        self.c = 1 / (self.c_b - self.c_a) * (
            (self.c_b - self.c) * self.alpha + (self.c - self.c_a) * self.beta)

    def scale_input(self):
        """Scales both the input values and the given data"""
        self.get_scaling_factors()
        self.scale_y0()
        self.scale_c()

    def scale_up_solution(self):
        """Scales up the solution based on the factors previously found
        for the given datapoints"""
        self.yps = 1 / (self.beta - self.alpha) * (
            (self.yps - self.alpha) * self.c_b + (self.beta - self.yps) * self.c_a)

    def printparameters(self):
        """Helperfunction for debugging"""
        print("K:", self.K)
        print("tau:", self.tau)
        print("d0", self.d0)
        print("I:", self.I)
        print("y0:", self.y0)
        print("c:", self.c)
        print("Z:", self.Z)
        print("yps:", self.yps)

    def get_P_km1(self, k):
        """One step when back propagating to find P"""
        return self.P[k + 1] + self.h * np.transpose(self.W[k]) @ (self.activation_function_derivated(self.W[k] @ self.Z[k] + self.b[k]) * self.P[k + 1])

    def dJ_dW(self):  # get dJ/dW which is element of theta
        dJ_dW = np.zeros(np.shape(self.W))
        for k in range(self.K - 1):
            dJ_dW[k] = self.h * (self.P[k + 1] * self.activation_function_derivated(
                self.W[k] @ self.Z[k] + self.b[k]) @ np.transpose(self.Z[k]))
        return dJ_dW

    def dJ_db(self):  # get dJ/db which is element of theta
        dJ_db = np.zeros(np.shape(self.b))
        for k in range(self.K - 1):
            dJ_db[k] = self.h * (self.P[k + 1] * self.activation_function_derivated(
                self.W[k] @ self.Z[k] + self.b[k])) @ np.ones((self.I, 1))

        return dJ_db

    def dJ_dw(self):  # get dJ/dw which is element of theta
        Z_K = self.Z[-1]
        return Z_K @ ((self.yps - self.c) * self.hypothesis_function_derivated(np.transpose(Z_K) @ self.w + self.mu * np.ones((self.I, 1))))

    def dJ_dmu(self):  # get dJ/dmu which is element of theta
        Z_K = self.Z[-1]
        return self.hypothesis_function_derivated(np.transpose(np.transpose(Z_K) @ self.w + self.mu * np.ones((self.I, 1))) @ (self.yps - self.c))

    def get_theta(self):
        """Returns all the weights, theta, componentwise"""
        return self.dJ_dW(), self.dJ_db(), self.dJ_dw(), self.dJ_dmu()

    def plot_cost(self):
        """Plots the cost function at each iteration"""
        plt.figure()
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.grid(True)
        plt.xlabel("Iterations")
        plt.ylabel(r'$J(\theta)$')

    def train_vanilla(self, iterations):
        """Training the model using the vanilla gradient method"""
        self.cost = np.zeros(iterations)  # Initialize storage for the cost

        for i in range(iterations):
            self.initialize_Z()
            self.initialize_yps()
            self.initialize_P()
            # Get theta values
            dJ_dW, dJ_db, dJ_dw, dJ_dmu = self.get_theta()
            self.W = simple_scheme(self.W, dJ_dW, self.tau)
            self.b = simple_scheme(self.b, dJ_db, self.tau)
            self.w = simple_scheme(self.w, dJ_dw, self.tau)
            self.mu = simple_scheme(self.mu, dJ_dmu, self.tau)

            self.cost[i] = self.objective_function()

    def train_adams_descent(self, iterations):
        """Training the model using the adams descent algorithm
        for optimization"""
        self.cost = np.zeros(iterations)

        m_mu = 0
        v_mu = 0
        m_w = np.zeros(np.shape(self.w))
        v_w = np.zeros(np.shape(self.w))
        m_W = np.zeros(np.shape(self.W))
        v_W = np.zeros(np.shape(self.W))
        m_b = np.zeros(np.shape(self.b))
        v_b = np.zeros(np.shape(self.b))

        for i in range(1, iterations + 1):
            self.initialize_Z()
            self.initialize_yps()
            self.initialize_P()
            # Get theta values
            dJ_dW, dJ_db, dJ_dw, dJ_dmu = self.get_theta()

            self.W, m_W, v_W = adam_descent_step(self.W, dJ_dW, i, m_W, v_W)
            self.b, m_b, v_b = adam_descent_step(self.b, dJ_db, i, m_b, v_b)
            self.w, m_w, v_w = adam_descent_step(self.w, dJ_dw, i, m_w, v_w)
            self.mu, m_mu, v_mu = adam_descent_step(
                self.mu, dJ_dmu, i, m_mu, v_mu)

            self.cost[i - 1] = self.objective_function()

    def evaluate_data(self, data):
        """Evaluate new data with our weights found during the
        training phase"""
        self.y0 = data
        self.scale_y0()
        self.embed()
        self.initialize_Z()
        self.initialize_yps()
        self.scale_up_solution()


# One step of the adam gradient decent for one parameter
def adam_descent_step(U, dU, j, m, v):
    beta_1 = 0.9
    beta_2 = 0.999
    alpha = 0.01
    eps = 10**(-8)
    g = dU
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * (g * g)
    m_hat = m / (1 - beta_1**j)
    v_hat = v / (1 - beta_2**j)
    U = U - alpha * m_hat / (np.sqrt(v_hat) + eps)
    return U, m, v


def simple_scheme(U, dU, tau):
    """One step of the vanilla gradient method to optimize
    weights and bias, for one parameter"""
    return U - tau * dU

