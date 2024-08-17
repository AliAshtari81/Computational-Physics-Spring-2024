%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time

t1 = time.time()

L = 10 # length of chain
bd = 5 # bond dimension
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
Sx_L, Sy_L, Sz_L, Sx_R, Sy_R, Sz_R = {}, {}, {}, {}, {}, {}
Id_L = Id(bd+1)
Id_R = Id(bd+1)
for i in range(1, bd+2):
    Sx_L[i] = sp.kron(Id(i-1), sp.kron((hbar/2)*sigma_x, Id(bd+1-i), format = 'csr'), format='csr')
    Sy_L[i] = sp.kron(Id(i-1), sp.kron((hbar/2)*sigma_y, Id(bd+1-i), format = 'csr'), format='csr')
    Sz_L[i] = sp.kron(Id(i-1), sp.kron((hbar/2)*sigma_z, Id(bd+1-i), format = 'csr'), format='csr')
    #Sx_R[i] = Sx_L[i].copy()
    #Sy_R[i] = Sy_L[i].copy()
    #Sz_R[i] = Sz_L[i].copy()
    Sx_R[i] = sp.kron(Id(bd+1-i), sp.kron((hbar/2)*sigma_x, Id(i-1), format = 'csr'), format='csr')
    Sy_R[i] = sp.kron(Id(bd+1-i), sp.kron((hbar/2)*sigma_y, Id(i-1), format = 'csr'), format='csr')
    Sz_R[i] = sp.kron(Id(bd+1-i), sp.kron((hbar/2)*sigma_z, Id(i-1), format = 'csr'), format='csr')
Sx_Sx = {}
Sz_Sz = {}
for i in range(1, bd+2):
    Sx_Sx[i-1] = Sx_L[1].dot(Sx_L[i])
    Sz_Sz[i-1] = Sz_L[1].dot(Sz_L[i])
# Hamiltonian operators
H_L = sp.csr_matrix((2**(bd+1), 2**(bd+1)))
for i in range(1, bd+1):
    H_L = H_L + Sx_L[i].dot(Sx_L[i+1]) + Sy_L[i].dot(Sy_L[i+1]) + Sz_L[i].dot(Sz_L[i+1])
H_R = sp.csr_matrix((2**(bd+1), 2**(bd+1)))
for i in range(bd+1, 1, -1):
    #H_R = H_R + Sx_R[i].dot(Sx_R[i+1]) + Sy_R[i].dot(Sy_R[i+1]) + Sz_R[i].dot(Sz_R[i+1])
    H_R = H_R + Sx_R[i].dot(Sx_R[i-1]) + Sy_R[i].dot(Sy_R[i-1]) + Sz_R[i].dot(Sz_R[i-1])
# Constructing super-block hamiltonian
H = sp.kron(H_L, Id_R, format='csr') + sp.kron(Id_L, H_R, format='csr') +\
    sp.kron(Sx_L[bd+1], Sx_R[bd+1], format='csr') + sp.kron(Sy_L[bd+1], Sy_R[bd+1], format='csr') +\
    sp.kron(Sz_L[bd+1], Sz_R[bd+1], format='csr')
H = (H + H.T.conj())/2
# Diagonalizing the hamiltoninan (Exact diagonalization)
E, U = sp.linalg.eigs(H, k=1, which='SR')
# Evaluating reduced density matrices
psi = U.reshape(2**(bd+1), 2**(bd+1))
rho_L = psi.dot(psi.T.conj())
rho_L = (rho_L + rho_L.T.conj())/2
rho_R = (psi.T).dot(psi.conj())
rho_R = (rho_R + rho_R.T.conj())/2
# Truncation
def truncation_matrix(M, n):
    w, u = sp.linalg.eigs(M, k=n, which='LR')
    indices = (-1*w).argsort()
    T = u[:, indices]
    return sp.csr_matrix(T)

# Start DMRG (Density Matrix Renormalization Group) process
T_L = truncation_matrix(rho_L, 2**bd)
T_R = truncation_matrix(rho_R, 2**bd)
for l in range(bd+2, L//2 + 1):
    for i in range(1, l):
        # Left block
        Sx_L_tilda = (T_L.T.conj()).dot(Sx_L[i].dot(T_L))
        Sx_L[i] = sp.kron(Sx_L_tilda, I, format='csr')
        Sy_L_tilda = (T_L.T.conj()).dot(Sy_L[i].dot(T_L))
        Sy_L[i] = sp.kron(Sy_L_tilda, I, format='csr')
        Sz_L_tilda = (T_L.T.conj()).dot(Sz_L[i].dot(T_L))
        Sz_L[i] = sp.kron(Sz_L_tilda, I, format='csr')
        # Right block
        Sx_R_tilda = (T_R.T.conj()).dot(Sx_R[i].dot(T_R))
        Sx_R[i] = sp.kron(Sx_R_tilda, I, format='csr')
        Sy_R_tilda = (T_R.T.conj()).dot(Sy_R[i].dot(T_R))
        Sy_R[i] = sp.kron(Sy_R_tilda, I, format='csr')
        Sz_R_tilda = (T_R.T.conj()).dot(Sz_R[i].dot(T_R))
        Sz_R[i] = sp.kron(Sz_R_tilda, I, format='csr')
        # Correlation functions
        Sx_1i_tilde = (T_L.T.conj()).dot(Sx_Sx[i-1].dot(T_L))
        Sx_Sx[i-1] = sp.kron(Sx_1i_tilde, I, format='csr')
        Sz_1i_tilde = (T_L.T.conj()).dot(Sz_Sz[i-1].dot(T_L))
        Sz_Sz[i-1] = sp.kron(Sz_1i_tilde, I, format='csr') 
    # Added site to the left
    Sx_L[l] = sp.kron(Id(bd), (hbar/2)*sigma_x, format='csr')
    Sy_L[l] = sp.kron(Id(bd), (hbar/2)*sigma_y, format='csr')
    Sz_L[l] = sp.kron(Id(bd), (hbar/2)*sigma_z, format='csr')    
    # Added site to the right
    Sx_R[l] = sp.kron((hbar/2)*sigma_x, Id(bd), format='csr')
    Sy_R[l] = sp.kron((hbar/2)*sigma_y, Id(bd), format='csr')
    Sz_R[l] = sp.kron((hbar/2)*sigma_z, Id(bd), format='csr')
    # Correlation function
    Sx_Sx[l-1] = Sx_L[1].dot(Sx_L[l])
    Sz_Sz[l-1] = Sz_L[1].dot(Sz_L[l])
    # Hamiltonian operators
    # Left block
    H_L_tilde = (T_L.T.conj()).dot(H_L.dot(T_L))
    H_L = sp.kron(H_L_tilde, I, format='csr') + sp.kron(Sx_L_tilda, (hbar/2)*sigma_x, format='csr')+\
            sp.kron(Sy_L_tilda, (hbar/2)*sigma_y, format='csr') +\
            sp.kron(Sz_L_tilda, (hbar/2)*sigma_z, format='csr')
    # Right block
    H_R_tilde = (T_R.T.conj()).dot(H_R.dot(T_R))
    H_R = sp.kron(H_R_tilde, I, format='csr') + sp.kron((hbar/2)*sigma_x, Sx_R_tilda, format='csr')+\
            sp.kron((hbar/2)*sigma_y, Sy_R_tilda, format='csr') +\
            sp.kron((hbar/2)*sigma_z, Sz_R_tilda, format='csr')
    # Super-block
    H = sp.kron(H_L, Id_R, format='csr') + sp.kron(Id_L, H_R, format='csr') +\
        sp.kron(Sx_L[l], Sx_R[l], format='csr') + sp.kron(Sy_L[l], Sy_R[l], format='csr') +\
        sp.kron(Sz_L[l], Sz_R[l], format='csr')
    H = (H + H.T.conj())/2
    # Diagonalizing the super-block hamiltonian
    E, U = sp.linalg.eigs(H, k=1, which = 'SR')
    # Evaluating reduced density matrices
    psi = U.reshape(2**(bd+1), 2**(bd+1))
    rho_L = psi.dot(psi.T.conj())
    rho_L = (rho_L + rho_L.T.conj())/2
    rho_R = (psi.T).dot(psi.conj())
    rho_R = (rho_R + rho_R.T.conj())/2
    # Truncation
    T_L = truncation_matrix(rho_L, 2**bd)
    T_R = truncation_matrix(rho_R, 2**bd)


print(f'Ground-state energy is {np.real(E[0])}')

# Average of <S_x> and <S_y>
Sx_avg = {}
Sy_avg = {}
for i in range(1, L//2 + 1):
    # Left block
    X_L = sp.kron(Sx_L[i], Id(bd+1), format='csr')
    Sx_avg[i] = (U.T.conj()).dot(X_L.dot(U))[0]
    Y = sp.kron(Sy_L[i], Id(bd+1), format='csr')
    Sy_avg[i] = (U.T.conj()).dot(Y.dot(U))[0]
    # Right block
    X_R = sp.kron(Id(bd+1), Sx_R[i], format='csr')
    Sx_avg[L-i+1] = (U.T.conj()).dot(X_R.dot(U))[0]
    Y_R = sp.kron(Id(bd+1), Sy_R[i], format='csr')
    Sy_avg[L-i+1] = (U.T.conj()).dot(Y_R.dot(U))[0]
    
print(f'average of <S_x> is {np.mean(list(Sx_avg.values()))}')
print(f'average of <S_y> is {np.mean(list(Sy_avg.values()))}')
plt.figure()
plt.plot(np.arange(1, L+1), Sx_avg.values(), 'o--')
plt.xlabel('i')
plt.ylabel(r'$\langle S_{x,i} \rangle$')
plt.title(r'average spin ($S_x$)')
plt.grid()
plt.show()
#
plt.figure()
plt.plot(np.arange(1, L+1), Sy_avg.values(), 'o--')
plt.xlabel('i')
plt.ylabel(r'$\langle S_{y,i} \rangle$')
plt.title(r'average spin ($S_y$)')
plt.grid()
plt.show()
# Correlation functions
def Tr(M):
    return np.sum(np.diag(M))
Corr_Sx_Sx = {}
Corr_Sz_Sz = {}
for i in range(1, L//2 + 1):
    # Sx_Sx
    #print(((rho_L).dot(Sx_Sx[i-1].toarray())))
    Corr_Sx_Sx[i-1] = Tr(((rho_L).dot(Sx_Sx[i-1].toarray()))) # both in left block
    Corr_Sx_Sx[L-i] = Tr((psi.T.conj()).dot(Sx_L[1].toarray().dot(psi.dot(Sx_R[i].toarray().T)))) # one in left and one in right block
    # Sz_Sz
    Corr_Sz_Sz[i-1] = Tr(((rho_L).dot(Sz_Sz[i-1].toarray()))) # both in left block
    Corr_Sz_Sz[L-i] = Tr((psi.T.conj()).dot(Sz_L[1].toarray().dot(psi.dot(Sz_R[i].toarray().T)))) # one in left and one in right block
#

plt.figure()
plt.plot(np.arange(0, L), list(Corr_Sx_Sx.values()), 'o--')
plt.xlabel('j-i')
plt.ylabel(r'$\langle S_{x,i} S_{x,j} \rangle$')
plt.title(r'Spin-Spin ($S_x,S_x$) correlation function')
plt.grid()
plt.show()
#
plt.figure()
plt.plot(np.arange(0, L), list(Corr_Sz_Sz.values()), 'o--')
plt.xlabel('j-i')
plt.ylabel(r'$\langle S_{z,i} S_{z,j} \rangle$')
plt.title(r'Spin-Spin ($S_z,S_z$) correlation function')
plt.grid()
plt.show()
#
t2 = time.time()
t = t2 - t1
print(f'Run time of the program is {t//60} min, {t%60} s.')
