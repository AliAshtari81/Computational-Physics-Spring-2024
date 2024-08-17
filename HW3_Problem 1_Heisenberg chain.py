#%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time # for evaluating the run time of the program
t_initial=time.time()
L=12
J1=-1.
J2=0.0
bnd=0 # periodic boundary condition
hbar=1
# Constructing spin matrices as sparse matrices
Id=sp.csr_matrix(np.array([[1., 0.], [0., 1.]]))
sigma_x=sp.csr_matrix(np.array([[0., 1.], [1., 0.]]))
sigma_y=sp.csr_matrix(np.array([[0., -1.j],[1.j, 0.]]))
sigma_z=sp.csr_matrix(np.array([[1., 0.],[0., -1.]]))
# Expanding the Hilbert space
Sx, Sy, Sz={}, {}, {}
Sx[1]=sp.kron((hbar/2)*sigma_x, sp.identity(2**(L-1)))
Sy[1]=sp.kron((hbar/2)*sigma_y, sp.identity(2**(L-1)))
Sz[1]=sp.kron((hbar/2)*sigma_z, sp.identity(2**(L-1)))
#
Sx[L]=sp.kron(sp.identity(2**(L-1)), (hbar/2)*sigma_x)
Sy[L]=sp.kron(sp.identity(2**(L-1)), (hbar/2)*sigma_y)
Sz[L]=sp.kron(sp.identity(2**(L-1)), (hbar/2)*sigma_z)
#
for i in range(2, L):
    Sx[i]=sp.kron(sp.identity(2**(i-1)), sp.kron((hbar/2)*sigma_x, sp.identity(2**(L-i))))
    Sy[i]=sp.kron(sp.identity(2**(i-1)), sp.kron((hbar/2)*sigma_y, sp.identity(2**(L-i))))
    Sz[i]=sp.kron(sp.identity(2**(i-1)), sp.kron((hbar/2)*sigma_z, sp.identity(2**(L-i))))
# Constructing hamiltonian
H=sp.csr_matrix((2**L, 2**L))
# Nearest neighbor interaction
for i in range(1, L):
    H+= J1*(Sx[i].dot(Sx[i+1]) + Sy[i].dot(Sy[i+1]) + Sz[i].dot(Sz[i+1]))
if bnd==1: # applying periodic boundary condition
    H+=J1*(Sx[L].dot(Sx[1]) + Sy[L].dot(Sy[1]) + Sz[L].dot(Sz[1]))
# Next nearest neighbor interaction
for i in range(1, L-1):
    H+= J2*(Sx[i].dot(Sx[i+2]) + Sy[i].dot(Sy[i+2]) + Sz[i].dot(Sz[i+2]))
if bnd==1: # applying periodic boundary condition
    H+= J2*(Sx[L-1].dot(Sx[1]) + Sy[L-1].dot(Sy[1]) + Sz[L-1].dot(Sz[1]) +\
            Sx[L].dot(Sx[2]) + Sy[L].dot(Sy[2]) + Sz[L].dot(Sz[2]))
# projecting hamiltonian in subspace with S_{z, tot}=0
Sz_tot=sp.csr_matrix((2**L, 2**L))
for i in range(1, L+1):
    Sz_tot += Sz[i]
Sz_tot_diag=Sz_tot.diagonal()
F=(Sz_tot_diag==0)
H_p=H[:, F][F, :] # projected hamiltonian

# Diagonalizing the projected hamiltonian
E, U=sp.linalg.eigs(H_p, k=20, which='SR')
S=E.argsort()
E=np.real(E[S])
U=U[:, S]
print('Energy eigenvalues are:')
print(E)
psi_p=U[:, 0] # ground state wave-function in the projected hilbert space
# Plotting energy eigenvalues
plt.figure()
plt.bar(np.arange(1, 21), E)
plt.xlabel('number of state')
plt.ylabel('energy eigenvalue')
labels=[f'{i}' for i in range(1, 21)]
plt.xticks(np.arange(1, 21), labels)
plt.show()
# Reconstructing wave-function in the original hilbert space to evaluate the expectation values
psi=np.zeros(2**L)
psi[F]=psi_p
psi=sp.csr_matrix(psi.reshape(-1, 1))

# Computing spin-spin correlation function
SS_Corr={}
for i in range(1, L+1):
    o=Sx[1].dot(Sx[i]) + Sy[1].dot(Sy[i]) + Sz[1].dot(Sz[i])
    SS_Corr[i]=np.real(((psi.T.conj()).dot(o.dot(psi))).toarray()[0])
print('Correlation function is:')
print(list(SS_Corr.values()))
# Plotting spin-spin correlation function
plt.figure()
plt.plot(SS_Corr.keys(), SS_Corr.values(), 'o--')
plt.xlabel('distance')
plt.ylabel('Spin-Spin correlation function')
plt.grid()
plt.show()
# Computing entanglement entropy
entanglement_entropy={}
for n in range(0, L+1):
    dim_L=2**n
    dim_R=2**(L-n)
    psi_tilda=psi.reshape(dim_R, dim_L)
    rho_L=((psi_tilda.T).dot(psi_tilda.conj())).toarray()
    p, v= np.linalg.eigh(rho_L)
    p[abs(p)<1e-12]=1e-12
    if n==0 or n==L:
        entanglement_entropy[n]=0
    else:
        entanglement_entropy[n]=-p.dot(np.log(p))
print('Entanglement entropy is: ')   
print(entanglement_entropy)
# Plotting the entanglement entropy
plt.figure()
plt.plot(entanglement_entropy.keys(), entanglement_entropy.values(), 'o--')
plt.xlabel('number of spins in the left hand side')
plt.ylabel('entanglement entropy')
plt.grid()
plt.show()
#
t_final=time.time()
print(f'Run time of the program is {t_final - t_initial} s')








