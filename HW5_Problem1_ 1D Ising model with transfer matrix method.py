#%matplotlib qt
# 1D Ising model with transfer matrix method
import numpy as np
import matplotlib.pyplot as plt
import time
t1 = time.time()
J = 0.3 # J/k_B T
h = 0.1 # h/k_B T
N=60
# defining the transfer matrix
T = np.array([[np.exp(J+h), np.exp(-J)],[np.exp(-J), np.exp(J-h)]])
w, v = np.linalg.eigh(T)
# Evaluating the partition function of the system
def partition_function(N):
    return w[0]**N + w[1]**N
Z = partition_function(N)
print(f'Z = {Z}')
# Evaluating magnetization density
sigma_z = np.array([[1., 0.],[0., -1.]])
def magnetization_density(N):
    return (w[0]**N * v[:, 0].T.dot(sigma_z.dot(v[:, 0])) + \
            w[1]**N * v[:, 1].T.dot(sigma_z.dot(v[:, 1])))/Z
m = magnetization_density(N)
print(f'm = {m}')
# Evaluating spin-spin correlation function and plotting it against the distance |j-i|
def spin_spin_correlation_function(N, i, j):
    return (w[0]**(N+i-j) * v[:, 0].T.dot(sigma_z.dot((np.linalg.matrix_power(T, j-i)).dot(sigma_z)).dot(v[:, 0])) +\
            w[1]**(N+i-j) * v[:, 1].T.dot(sigma_z.dot((np.linalg.matrix_power(T, j-i)).dot(sigma_z)).dot(v[:, 1])))/Z
SS_Corr=[]
for j in range(1, N+2):
    SS_Corr.append(spin_spin_correlation_function(N, 1, j))
indices=[i for i in range(0, N+1)]
plt.plot(indices, SS_Corr, 'o-', markersize=3)
plt.xlabel('distance |j-i|')
plt.ylabel(r'$\langle S_i S_j \rangle$')
plt.title('Spin-Spin Correlation function')
plt.grid()
t2 = time.time()
print(f'Run time of the program is {t2 - t1} s .')