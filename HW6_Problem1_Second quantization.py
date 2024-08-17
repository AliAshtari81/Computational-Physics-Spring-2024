%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time

t1 = time.time()
L = 8 # number of sites
N = 3 # number of electrons
t = 1.0
V = 1.0
# constructing creation and annihilation operators
sigma_z = sp.csr_matrix(np.array([[1., 0.],[0., -1.]]))
sigma_p = sp.csr_matrix(np.array([[0., 1.],[0., 0.]]))
def Id(n):
    '''
    Identity operator for n sites.
    '''
    if n==0:
        return 1.
    else:
        return sp.identity(2**n, format = 'csr')
        
c_dagger = []
Sz = 1.
for i in range(L):
    c_dagger.append(sp.kron(Sz, sp.kron(sigma_p, Id(L-i-1), format='csr'), format='csr'))
    Sz = sp.kron(Sz, sigma_z, format = 'csr')

# creating number density operatios
n0 = sp.csr_matrix(np.array([[1., 0.],[0., 0.]]))
n = []
for i in range(L):
    n.append(sp.kron(Id(i), sp.kron(n0, Id(L-i-1), format='csr'), format='csr'))
# Constructing the hamiltonian operator in second quantization
def H_second_quantization():
    H = sp.csr_matrix(n[0].shape)
    for i in range(L):
        X = c_dagger[np.mod(i+1, L)].dot(c_dagger[i].T.conj())
        H = H - t * (X + X.T.conj()) + V * (n[i].dot(n[np.mod(i+1, L)]))
    return H
H_SQ = H_second_quantization()
# Projecting the hamiltonian in subspace with N = 3 electrons (fermions)
N_tot = sp.csr_matrix(n[0].shape)
for i in range(L):
    N_tot += n[i]
N_tot_diag = N_tot.diagonal()
F = (N_tot_diag == N)
H_SQ_eff = H_SQ[:, F][F, :]
# Diagonalizing the hamiltonian
E_SQ, U_SQ = sp.linalg.eigs(H_SQ_eff, k=3, which = 'SR')
indices = E_SQ.argsort()
E_SQ = np.real(E_SQ[indices])
U_SQ = U_SQ[:, indices]
# Ground-state energy
print(f'Ground-state energy is {E_SQ[0]}')
# Ground-state wave-function
psi_p = U_SQ[:, 0] # in projected Hilbert space
psi = np.zeros(2**L)
psi[F] = psi_p
psi = psi.reshape(-1, 1) # in the original Hilbert space
#
plt.figure()
plt.plot(psi)
plt.grid()
plt.title('Ground-state wave-function')
plt.ylabel(r'$\psi$')
plt.xlabel('index in the vector')
plt.show()
# density-density correlation function with ground-state wave function
Corr = []
for i in range(L+1):
    n_ij = n[0].dot(n[np.mod(i, L)])
    Corr.append(np.real(((psi.T.conj()).dot(n_ij.dot(psi)))[0]))

plt.figure()
plt.plot(np.arange(0, L+1), Corr, 'o--')
plt.grid()
plt.xlabel('j-i')
plt.ylabel(r'$\langle n_i n_j \rangle$')
plt.title('real part of density-density correlation function')
plt.show()
#
t2 = time.time()
print(f'Run time of the program is {t2 - t1} s')