#%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import time

t1 = time.time()
J = 0.3 # J/k_B T
h = 0.1 # h/k_B T
M = 4
N = 60
# defining the "2^m by 2^m" transfer matrix
spins=[+1, -1]
def f(S1, S2, J, h):
    F=0
    for j in range(M):
        F = F + J*(S1[j]*S2[j] + 0.5*(S1[j]*S1[np.mod(j+1, M)] + S2[j]*S2[np.mod(j+1, M)])) +\
                0.5*h*(S1[j] + S2[j])
    return np.exp(F)
S1 = []
for i1 in spins:
    for i2 in spins:
        for i3 in spins:
            for i4 in spins:
                S1.append([i1, i2, i3, i4])
S2 = S1.copy()

def transfer_matrix(M, N, J, h):
    T = np.zeros((2**M, 2**M)) 
    for i in range(2**M):
        for j in range(2**M):
            T[i, j] = f(S1[i], S2[j], J, h)
    return T
T_1 = transfer_matrix(M, N, J, h)
T_2 = transfer_matrix(M, N, J, h+0.0001)
# Computing the partition function
def partition_function(T, M, N):
    w, U = np.linalg.eigh(T)
    return np.sum(w**N)
Z_1 = partition_function(T_1, M, N)
Z_2 = partition_function(T_2, M, N)
print(f'Z = {Z_1}')
#print(Z_2)
# Computing magnetization density
def magnetization_by_differentiation(Z1, Z2, delta_h):
    return (np.log(Z2) - np.log(Z1))/(delta_h * M * N)
m_1 = magnetization_by_differentiation(Z_1, Z_2, delta_h=0.0001)
print(f'm = {m_1} (By evaluating derivative of ln(Z) with respect to h/ (k_B T))')
# Defining spin matrices
Sz_1 = np.diag([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])
Sz_2 = np.diag([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1])
Sz_3 = np.diag([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
Sz_4 = np.diag([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
S = [Sz_1, Sz_2, Sz_3, Sz_4]
#
def magnetization_by_trace(M, N, w, U, Z): # w and U are the set of eigenvalues and eigenfunctions of transfer matrix 
    X = 0
    Sz = Sz_1 + Sz_2 + Sz_3 + Sz_4
    for i in range(2**M):
        X += w[i]**N * (U[:, i].T.dot(Sz.dot(U[:, i])))
    return X/(4*Z)
w1, U1 = np.linalg.eigh(T_1)
m_1_t = magnetization_by_trace(M, N, w1, U1, Z_1)
print(f'm = {m_1_t} (By evaluating trace of the corresponding matrix)')
# Computing correlation function
def correlation_function(T, Z, M, N, w, U):
    Corr = np.zeros((M+1, N+1))
    for i in range(0, N+1):
        for j in range(0, M+1):
            x = 0
            for l in range(2**M):
                X = S[0].dot(np.linalg.matrix_power(T, i).dot(S[np.mod(j, M)]))
                x += (w[l]**(N-i) * (U[:, l].T.dot(X.dot(U[:, l]))))
            Corr[j, i] = x
    return Corr/Z
Corr = correlation_function(T_1, Z_1, M, N, w1, U1)
# Plotting the correlation function
plt.figure()
plt.imshow(Corr, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.xlabel('i')
plt.ylabel('j')
plt.title('Heatmap of spin-spin correlation function')
plt.show()
#
X, Y = np.meshgrid(np.arange(N+1), np.arange(M+1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Corr, cmap='viridis')
ax.set_xlabel('i')
ax.set_ylabel('j')
ax.set_zlabel(r'$\langle S_i S_j \rangle$')
ax.set_title('3D plot of spin-spin correlation function')
plt.show()

#
t2 = time.time()
t = t2 - t1
print(f'Run time of the program is {t//60} min, {t%60} s.')