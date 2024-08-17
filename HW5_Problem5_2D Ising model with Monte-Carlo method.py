#%matplotlib qt
# 2D Ising model with Monte-Carlo method
import numpy as np
import matplotlib.pyplot as plt
import time
# importing functions for finding transition tempreture
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

t1 = time.time()
N=20
M=20

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
# Defining functions to evaluate the desired expectation values
def magnetization_density(S, M, N):
    return (np.sum(S))/(M*N)
def spin_spin_correlation_function(S, M, N, i, j):
    return (np.sum(S * np.roll(S, (i, j), axis=(0, 1))))/(M*N)


def Metropolis(S, M, N, L, J, h, compute_correlation):
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
        if k>(L - L//10) and (compute_correlation==True):
            for i1 in range(M+1):
                for i2 in range(N+1):
                    Corr[i1, i2] = Corr[i1, i2] + spin_spin_correlation_function(S, M, N, i1, i2)
    Corr = Corr/(L//10)
            
    return (m, E, Corr)
            


def plot_magnetization(BJ_list, m_list):
    '''
    Plotting magnetization density against tempreture.
    '''
    plt.figure()
    plt.plot(BJ_list, m_list, 'o--', markersize=3)
    plt.grid()
    plt.xlabel(r'$J / k_B T$')
    plt.ylabel('m')
    plt.title(r'Magnetization density against $\beta J = J / k_B T$')
    plt.show()
    

#J=0.7
#plot_energy_spin_per_step(1000000, M, N, J, h)
#t2 = time.time()
#t = t2 - t1
#print(f'Run time of the program is {t//60} min, {t%60} s.')

S0 = initial(M, N)
L=500000
J=0.3
h=0.0
m, E, C = Metropolis(S0, M, N, L, J, h, compute_correlation=True)
plt.figure(1)
plt.plot(np.arange(L+1), E)
plt.grid()
plt.xlabel('steps')
plt.ylabel('E')
plt.title(r'Energy against time steps for $\beta J=0.3$ and $\beta h = 0.0$ ')
plt.show()
#
plt.figure(2)
plt.plot(np.arange(L+1), m)
plt.grid()
plt.xlabel('steps')
plt.ylabel('m')
plt.title(r'Magnetization density against time steps for $\beta J=0.3$ and $\beta h = 0.0$')
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

# Plotting magnetization agianst tempreture
BJ_list = list(np.linspace(0.1, 0.8, 80))
m_list = []
for BJ in BJ_list:
    S = initial(M, N)
    m_BJ, E_BJ, C_BJ = Metropolis(S, M, N, L, BJ, h, compute_correlation=False)
    m_list.append(np.mean(m_BJ[-(L//10):]))
plot_magnetization(BJ_list, m_list)

# Find transition tempreture

#f = interp1d(BJ_list, m_list, kind='cubic')
a = np.mean([np.mean(m_list[-5:]), np.mean(m_list[:5])])
#BJ_c = fsolve(lambda x: f(x) - a, 0.47)
#print(f'transition tempreture B_c J = J/(k_B T_c) = {BJ_c}')
X = np.where(abs(m_list - a)<0.3)
L=[]
for i in X[0]:
    L.append(BJ_list[i])

print(f'transition tempreture B_c J = J/(k_B T_c) = {np.mean(L)}')
#
t2 = time.time()
t = t2 - t1
print(f'Run time of the program is {t//60} min, {t%60} s.')