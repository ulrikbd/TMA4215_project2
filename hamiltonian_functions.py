import numpy as np

class NonlinearPendulum():
    """Hamiltonian formulation of the nonlinear pendulum problem"""
    def __init__(self, m, l, g=9.81):
        self.m = m
        self.l = l
        self.g = g

    def T(self, p):
        return 1/2 * p**2

    def V(self, q):
        return self.m * self.l * self.g * (1 - np.cos(q))

    def dT_dp(self, p):
        return p

    def dV_dq(self, q):
        return self.m * self.l * self.g * np.sin(q)


class HenonHeiles():
    """Hamiltonian formulation of the Henon_Heiles problem"""

    def T(self, p):
        return 1/2 * (p[0]**2 + p[1]**2)

    def V(self, q):
        return 1/2 * (q[0]**2 + q[1]**2) + q[0]**2 * q[1] - 1/3 * q[1]**2

    def dT_dp(self, p):
        return p

    def dV_dq(self, q):
        return np.array([q[0] + 2*q[0]*q[1],
            q[1] + q[0]**2 - q[1]**2])