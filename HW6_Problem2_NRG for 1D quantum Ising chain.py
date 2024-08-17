%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time

t1 = time.time()

L = 10 # length of chain
bd = 6 # bond dimension
hbar = 1
sigma_x = sp.csr_matrix(np.array([[0., 1.],[1., 0.]]))
sigma_y = sp.csr_matrix(np.array([[0., -1.j],[1.j, 0.]]))
sigma_z = sp.csr_matrix(np.array([[1., 0.],[0., -1.]]))
I = sp.identity(2, format='csr')
# Constructing the exact operators for bd+1 sites
def Id(n):
    '''
    Identity operator for n sites
    '''
    if n==0:
        return 1.
    else:
        return sp.identity(2**n, format='csr')
# Spin operators
Sx, Sy, Sz = {}, {}, {}
for i in range(1, bd+2):
    Sx[i] = sp.kron(Id(i-1), sp.kron((hbar/2)*sigma_x, Id(bd+1 - i), format='csr'), format='csr')
    Sy[i] = sp.kron(Id(i-1), sp.kron((hbar/2)*sigma_y, Id(bd+1 - i), format='csr'), format='csr')
    Sz[i] = sp.kron(Id(i-1), sp.kron((hbar/2)*sigma_z, Id(bd+1 - i), format='csr'), format='csr')
# Hamiltonian operator
H = sp.csr_matrix((2**(bd+1), 2**(bd+1)))
for i in range(1, bd+1):
    H = H + Sx[i].dot(Sx[i+1]) + Sy[i].dot(Sy[i+1]) + Sz[i].dot(Sz[i+1])
H = (H + H.T.conj())/2
E, T = sp.linalg.eigs(H, k=2**bd, which='SR')
indices = E.argsort()
E = np.real(E[indices])
T = sp.csr_matrix(T[:, indices])

Corr_Sx_Sx = []
Corr_Sz_Sz = []
for i in range(1, bd+2):
    Corr_Sx_Sx.append(Sx[1].dot(Sx[i]))
    Corr_Sz_Sz.append(Sz[1].dot(Sz[i]))
# Start NRG (Numerical Renormalization Group) process
for n in range(bd+2, L+1):
    # Spin operators
    for i in range(1, n): # truncation of spin operators
        #print(T.shape, Sx[i].shape, n, i)
        Sx_tilda = (T.T.conj()).dot(Sx[i].dot(T))
        Sx[i] = sp.kron(Sx_tilda, I, format='csr')
        Sy_tilda = (T.T.conj()).dot(Sy[i].dot(T))
        Sy[i] = sp.kron(Sy_tilda, I, format='csr')
        Sz_tilda = (T.T.conj()).dot(Sz[i].dot(T))
        Sz[i] = sp.kron(Sz_tilda, I, format='csr')
        #print(Sx[i].shape, Corr_Sx_Sx[i-1].shape, T.shape, i)
        # Truncation of correlation functions
        Sx_1i_tilda = (T.T.conj()).dot(Corr_Sx_Sx[i-1].dot(T))
        Corr_Sx_Sx[i-1] = sp.kron(Sx_1i_tilda, I, format='csr')
        Sz_1i_tilda = (T.T.conj()).dot(Corr_Sz_Sz[i-1].dot(T))
        Corr_Sz_Sz[i-1] = sp.kron(Sz_1i_tilda, I, format='csr')
    Sx[n] = sp.kron(Id(bd), (hbar/2)*sigma_x, format='csr')
    Sy[n] = sp.kron(Id(bd), (hbar/2)*sigma_y, format='csr')
    Sz[n] = sp.kron(Id(bd), (hbar/2)*sigma_z, format='csr')
    Corr_Sx_Sx.append(Sx[1].dot(Sx[n]))
    Corr_Sz_Sz.append(Sz[1].dot(Sz[n]))
    # Hamiltonian operator
    H_tilda = (T.T.conj()).dot(H.dot(T)) # truncation of hamiltonian operator
    H = sp.kron(H_tilda, I) + sp.kron(Sx_tilda, (hbar/2)*sigma_x, format='csr') +\
        sp.kron(Sy_tilda, (hbar/2)*sigma_y, format='csr') +\
        sp.kron(Sz_tilda, (hbar/2)*sigma_z, format='csr')
    H = (H + H.T.conj())/2
    # Diagonalization of new hamiltonian
    E, T = sp.linalg.eigs(H, k=2**bd, which = 'SR')
    indices = E.argsort()
    E = np.real(E[indices])
    T = sp.csr_matrix(T[:, indices])
    
print(f'Ground-state energy is {E[0]}')
# Expectation value <S_x> and <S_y>
psi = T.toarray()[:, 0] # ground-state wave-function
def average_spin(S, v):
    '''
    Return the average of expectation values of S with state vector v.
    '''
    S_list = []
    for i in range(1, L+1):
        S_list.append(np.real((v.T.conj()).dot(S[i].dot(v))))
    return S_list

Sx_list = average_spin(Sx, psi)
print(f'average of <S_x> is {np.mean(Sx_list)}')
plt.figure()
plt.plot(np.arange(1, L+1), Sx_list, 'o--')
plt.xlabel('i')
plt.ylabel(r'$\langle S_{x,i} \rangle$')
plt.title(r'average spin ($S_x$)')
plt.grid()
plt.show()
#
Sy_list = average_spin(Sy, psi)
print(f'average of <S_y> is {np.mean(Sy_list)}')
plt.figure()
plt.plot(np.arange(1, L+1), Sy_list, 'o--')
plt.xlabel('i')
plt.ylabel(r'$\langle S_{y,i} \rangle$')
plt.title(r'average spin ($S_y$)')
plt.grid()
plt.show()


# Spin-spin correlation functions
for i in range(L):
    Corr_Sx_Sx[i] = (psi.T.conj()).dot(Corr_Sx_Sx[i].dot(psi))
    Corr_Sz_Sz[i] = (psi.T.conj()).dot(Corr_Sz_Sz[i].dot(psi))
plt.figure()
plt.plot(np.arange(0, L), Corr_Sx_Sx, 'o--')
plt.xlabel('j-i')
plt.ylabel(r'$\langle S_{x,i} S_{x,j} \rangle$')
plt.title(r'Spin-Spin ($S_x,S_x$) correlation function')
plt.grid()
plt.show()
#
plt.figure()
plt.plot(np.arange(0, L), Corr_Sz_Sz, 'o--')
plt.xlabel('j-i')
plt.ylabel(r'$\langle S_{z,i} S_{z,j} \rangle$')
plt.title(r'Spin-Spin ($S_z,S_z$) correlation function')
plt.grid()
plt.show()
    
#
t2 = time.time()
t = t2 - t1
print(f'Run time of the program is {t//60} min, {t%60} s .')