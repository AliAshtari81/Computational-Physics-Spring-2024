#%matplotlib qt
# 2D Ising model with Monte-Carlo method
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

t1 = time.time()
J = 0.3 # J/k_B T
h = 0.1 # h/k_B T
N=60
M=4

def initial(M, N):
    '''
    Constructing random initial configuration.
    '''
    L = np.random.rand(M, N)
    return np.where(L>0.25, 1, -1)
def energy(S, M, N, J, h):
    '''
    Evaluate energy of a spin configuration
    '''
    E = 0
    for j in range(N):
        for i in range(M):
            E = E - J * S[i, j] * (S[i, np.mod(j+1, N)] + S[np.mod(i+1, M), j]) - h * S[i, j]
    return E
def energy_difference(S, M, N, i, j, J, h):
    '''
    Evaluate energy difference if we flip the spin in site i, j .
    '''
    return 2*J*S[i, j]*(S[i, np.mod(j+1, N)] + S[np.mod(i+1, M), j] +\
                        S[i, np.mod(j-1, N)] + S[np.mod(i-1, M), j]) + 2*h*S[i, j]
def Metropolis(S, M, N, L, J, h):
    '''
    A Metropolis algorithm with L iterations. 
    '''
    m=[magnetization_density(S, M, N)]
    E=[energy(S, M, N, J, h)]
    Corr = np.zeros((M+1, N+1))
    for k in range(L):
        i = np.random.randint(0, M)
        j = np.random.randint(0, N)
        delta_E = energy_difference(S, M, N, i, j, J, h)
        r = np.exp(-delta_E)
        if r > np.random.rand(): # Flip the spin
            E.append(E[k] + delta_E)
            m.append(m[k] - 2*S[i, j]/(M*N))
            S[i, j] = -1 * S[i, j]
        else: # Don't flip the spin
            E.append(E[k])
            m.append(m[k])
        if k>(L - L//10):
            for i1 in range(M+1):
                for i2 in range(N+1):
                    Corr[i1, i2] = Corr[i1, i2] + spin_spin_correlation_function(S, M, N, i1, i2)
    Corr = Corr/(L//10)
            
    return (m, E, Corr)
# Defining functions to evaluate the desired expectation values
def magnetization_density(S, M, N):
    return (np.sum(S))/(M*N)
def spin_spin_correlation_function(S, M, N, i, j):
    return (np.sum(S * np.roll(S, (i, j), axis=(0, 1))))/(M*N)
# Starting Markov-chains
S0 = initial(M, N)
L=500000
m, E, C = Metropolis(S0, M, N, L, J, h)
m_avg = np.mean(m[-(L//10):])
print(f'average m is {m_avg}')
#
plt.figure(1)
plt.plot(np.arange(L+1), E)
plt.grid()
plt.title('Energy against time steps')
plt.xlabel('steps')
plt.ylabel('E')
plt.show()
#
plt.figure(2)
plt.plot(np.arange(L+1), m)
plt.grid()
plt.title('Magnetization density against time steps')
plt.xlabel('steps')
plt.ylabel('m')
plt.show()
#
# Plotting spin-spin correlation function as a heatmap
plt.figure()
plt.imshow(C, cmap = 'viridis', interpolation='nearest')
plt.colorbar()
plt.xlabel('j')
plt.ylabel('i')
plt.title('Heatmap of spin-spin correlation function')
#
X, Y = np.meshgrid(np.arange(N+1), np.arange(M+1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, C, cmap='viridis')
ax.set_xlabel('j')
ax.set_ylabel('i')
ax.set_zlabel(r'$\langle S_i S_j \rangle$')
ax.set_title('3D plot of spin-spin correlation function')
plt.show()
#
t2 = time.time()
t = t2 - t1
print(f'Run time of the program is {t//60} min, {t%60} s.')