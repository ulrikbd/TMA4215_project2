from NeuralNetwork import NeuralNetwork, adam_descent_step, simple_scheme, sympletic_euler_step, stormer_verlet_step
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from ast import literal_eval
import re
from hamiltonian_functions import NonlinearPendulum, HenonHeiles

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
    data = concatenate(0, 20)
    y0 = data["P"]
    d = 9
    I = len(y0[1])
    iterations = 500
    K = 15
    h = 0.1
    tau = 0.08
    chunk_size = 500
    c = data["T"]
    c = c.reshape((I, 1))
    T = NeuralNetwork(K, tau, h, y0, d, c, I)
    T.train_stochastic_gradient_descent(iterations, chunk_size)
    T.plot_cost()

    test_data = generate_data(batch=24)
    T.evaluate_data(test_data["P"])
    residual = T.get_average_residual(test_data["T"])
    print(residual)
    plt.figure()
    plt.plot(test_data["t"], test_data["T"])
    plt.plot(test_data["t"], T.yps, 'r.')
    plt.show()



def plot_hamiltionian():
    data = generate_data(23)
    t = data["t"]
    T = data["T"]
    V = data["V"]
    plt.plot(t, T, label=r'$T(t)$')
    plt.plot(t, V, label=r'$V(t)$')
    plt.xlabel("t")
    plt.legend()
    plt.show()

def plot_hamiltonian_position():
    data = generate_data(0)
    t = data["t"]
    p = data["P"]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(p[0], p[1], p[2])
    plt.show()

def test_sympletic_euler_pendumlum():
    H = NonlinearPendulum(m=2, l=5)
    h = 0.001
    t = np.arange(0, 20, h)
    p = np.zeros(len(t))
    q = np.zeros(len(t))
    p[0] = 0
    q[0] = np.pi / 4
    for n in range(1, len(t)):
        q[n], p[n] = sympletic_euler_step(q[n-1], p[n-1], H.dT_dp, H.dV_dq, h)
    plt.figure()
    plt.plot(t, p, '.', label=r'$p, [p]=m/s$')
    plt.plot(t, q, '.', label=r'$q,[q]=rad$')
    plt.xlabel("t")
    plt.legend()
    plt.savefig("./plots/nonlinear_pendulum_sympletic_euler.pdf", bbox_inches="tight")


    fig = plt.figure()
    ax = plt.axes(projection= "3d")
    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'$q$')
    H_values = H.V(q) + H.T(p)
    plt.plot(p, q, H_values, '.', label=r'$H(q,p)$')
    plt.legend()
    plt.savefig("./plots/nonlinear_pendulum_sympletic_euler_hamiltonian.pdf", bbox_inches="tight")
    print((max(H_values)-min(H_values))/max(H_values))
    plt.show()

def test_stormer_verner_henon_heiles():
    H = HenonHeiles()
    h = 0.001
    t = np.arange(0, 20, h)
    p = np.zeros(shape=(2, len(t)))
    q = np.zeros(shape=(2, len(t)))
    p[:, 0] = np.array([0.1, 0.1])
    q[:, 0] = np.array([0, 0])
    for n in range(1, len(t)):
        q[:, n], p[:, n] = stormer_verlet_step(
            q[:, n-1], p[:, n-1], H.dT_dp, H.dV_dq, h)


    H_values = H.V(q) + H.T(p)
    H_avg = np.mean(H_values)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plt.plot(q[0], q[1], p[0], label=r'$p_1, \quad \dot{x}$')
    plt.plot(q[0], q[1], p[1], label=r'$p_2, \quad \dot{y}$')
    ax.set_xlabel(r'$q_1, \quad x$')
    ax.set_ylabel(r'$q_2, \quad y$')
    ax.set_title(r'$H=$' + str(round(H_avg, 4)))
    plt.legend()
    plt.savefig("./plots/henon_heiles_sv_low_energy.pdf", bbox_inches="tight")
    plt.show()

    H = HenonHeiles()
    h = 0.001
    t = np.arange(0, 20, h)
    p = np.zeros(shape=(2, len(t)))
    q = np.zeros(shape=(2, len(t)))
    p[:, 0] = np.array([0.4, 0.4])
    q[:, 0] = np.array([0, 0])
    for n in range(1, len(t)):
        q[:, n], p[:, n] = stormer_verlet_step(
            q[:, n-1], p[:, n-1], H.dT_dp, H.dV_dq, h)


    H_values = H.V(q) + H.T(p)
    H_avg = np.mean(H_values)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plt.plot(q[0], q[1], p[0], label=r'$p_1, \quad \dot{x}$')
    plt.plot(q[0], q[1], p[1], label=r'$p_2, \quad \dot{y}$')
    ax.set_xlabel(r'$q_1, \quad x$')
    ax.set_ylabel(r'$q_2, \quad y$')
    ax.set_title(r'$H=$' + str(round(H_avg, 4)))
    plt.legend()
    plt.savefig("./plots/henon_heiles_sv_high_energy.pdf", bbox_inches="tight")
    plt.show()

def main():
    get_T()


if __name__ == "__main__":
    main()
