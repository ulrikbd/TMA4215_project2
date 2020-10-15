from NeuralNetwork import NeuralNetwork, adam_descent_step, simple_scheme
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from ast import literal_eval
import re

plt.style.use('seaborn')
"""
Both of the following functions import data. The output of both functions are a dictionary containing 5 arrays
    t: the array of av time points
    Q: the position values (q)
    P: the momentum values (p)
    T: the kinetic energy
    V: the potential energy

The data files contain data from 50 different trajectories, i.e. simulation of the path for a point with some
initial position q0 and momentum p0.

The function generate_data gives you the data from one of these data files, while the function concatenate
gives you the data from multiple trajectories at once. The default arguments of concatenate give you all the data
alltogether.

The folder project_2_trajectories must be placed in the same folder as your program to work. If the folder is in
some other location, the path for this location can be put into the string start_path.
"""


def generate_data(batch=0):

    start_path = ""
    path = start_path + "project_2_trajectories/datalist_batch_" + \
        str(batch) + ".csv"
    with open(path, newline="\n") as file:
        reader = csv.reader(file)
        datalist = list(reader)

    N = len(datalist)
    t_data = np.array([float(datalist[i][0]) for i in range(1, N)])
    Q1_data = [float(datalist[i][1]) for i in range(1, N)]
    Q2_data = [float(datalist[i][2]) for i in range(1, N)]
    Q3_data = [float(datalist[i][3]) for i in range(1, N)]
    P1_data = [float(datalist[i][4]) for i in range(1, N)]
    P2_data = [float(datalist[i][5]) for i in range(1, N)]
    P3_data = [float(datalist[i][6]) for i in range(1, N)]
    T_data = np.array([float(datalist[i][7]) for i in range(1, N)])
    V_data = np.array([float(datalist[i][8]) for i in range(1, N)])

    Q_data = np.transpose(
        np.array([[Q1_data[i], Q2_data[i], Q3_data[i]] for i in range(N - 1)]))
    P_data = np.transpose(
        np.array([[P1_data[i], P2_data[i], P3_data[i]] for i in range(N - 1)]))

    return {"t": t_data, "Q": Q_data, "P": P_data, "T": T_data, "V": V_data}


def concatenate(batchmin=0, batchmax=50):
    dictlist = []
    for i in range(batchmin, batchmax):
        dictlist.append(generate_data(batch=i))
    Q_data = dictlist[0]["Q"]
    P_data = dictlist[0]["P"]
    T0 = dictlist[0]["T"]
    V0 = dictlist[0]["V"]
    tlist = dictlist[0]["t"]
    for j in range(batchmax - 1):
        Q_data = np.hstack((Q_data, dictlist[j + 1]["Q"]))
        P_data = np.hstack((P_data, dictlist[j + 1]["P"]))
        T0 = np.hstack((T0, dictlist[j + 1]["T"]))
        V0 = np.hstack((V0, dictlist[j + 1]["V"]))
        tlist = np.hstack((tlist, dictlist[j + 1]["t"]))
    return {"t": tlist, "Q": Q_data, "P": P_data, "T": T0, "V": V0}


data = generate_data()
print(np.shape(data["t"]))
print(np.shape(data["Q"]))
print(np.shape(data["P"]))
print(np.shape(data["T"]))
print(np.shape(data["V"]))


def get_T():
    data = concatenate(0, 1)
    y0 = data["P"]
    d = 6
    I = len(y0[1])
    iterations = 200
    K = 12
    h = 0.08
    tau = 0.08
    c = data["T"]
    c = c.reshape((I, 1))

    T = NeuralNetwork(K, tau, h, y0, d, c, I)
    T.train_adams_descent(iterations)
    T.plot_cost()
    plt.show()
    print(T.cost[-1] / I)


get_T()
