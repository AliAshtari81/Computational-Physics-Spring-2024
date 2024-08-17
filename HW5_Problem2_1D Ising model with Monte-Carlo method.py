#%matplotlib qt
# 1D Ising model with Monte-Carlo method
import numpy as np
import matplotlib.pyplot as plt
import time
t1 = time.time()
J = 0.3 # J/k_B T
h = 0.1 # h/k_B T
N=60

def initial(N):
    '''
    Constructing random initial configuration.
    '''
    L= np.random.rand(N)
    return np.where(L>0.25, 1, -1)
def energy(S, N, J, h):
    '''
    Evaluate energy of a spin configuration
    '''
    E=0
    for i in range(N):
        E = E - J*S[i]*S[np.mod(i+1, N)] - h * S[i]
    return E
def energy_difference(S, N, i, J, h):
    '''
    Evaluate energy difference if we flip the spin in site i.
    '''
    return 2*J*S[i]*(S[np.mod(i-1, N)] + S[np.mod(i+1, N)]) + 2*h*S[i]

def Markov_chain(S, N, L, J, h):
    '''
    A markov-chain with length L and random initial point 
    '''
    initial = np.random.randint(0, N)
    for i in range(L):
        j = np.mod(initial + i, N)
        delta_E = energy_difference(S, N, j, J, h)
        r = np.exp(-delta_E)
        if r > np.random.rand():
            S[j] = -1 * S[j]
# Defining functions to evaluate the desired expectation values
def magnetization_density(S, N):
    return (np.sum(S))/N
def spin_spin_correlation_function(S, N, j):
    return (np.dot(S, np.roll(S, j)))/N
# Starting Markov-chains
chains = 300 # number of Markov-chains
m=[]
SS_Corr={}
for i in range(0, N+1):
    SS_Corr[i]=[]
for k in range(chains):
    S = initial(N)
    Markov_chain(S, N, L=300000, J=J, h=h)
    m.append(magnetization_density(S, N))
    for j in range(0, N+1):
        SS_Corr[j].append(spin_spin_correlation_function(S, N, j))
m_avg = np.mean(m)
print(f'm = {m_avg}')
SS_Corr_mean=[]
for j in range(0, N+1):
    SS_Corr_mean.append(np.mean(SS_Corr[j]))
distance = [j for j in range(0, N+1)]
plt.plot(distance, SS_Corr_mean, 'o-', markersize=3)
plt.grid()
plt.xlabel('distance |j-i|')
plt.ylabel(r'$\langle S_i S_j \rangle$')
plt.title('Spin-Spin Correlation function')
plt.show()

#
t2 = time.time()
print(f'Run time ot the program is {t2 - t1} s .')
    