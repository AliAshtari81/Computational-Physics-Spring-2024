#%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time # for evaluating the run time of the program
t_initial=time.time()
L=12
a=3 # vertical length
b=4 # horizontal length
J1=1.
J2=0.3
bnd=1 # periodic boundary condition
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

#
def ij(i, j): # relabing points
    return (i-1)*a + j
# Constructing the hamiltonian
H=sp.csr_matrix((2**L, 2**L))
# Nearest neighbor interaction
for i in range(1, b+1):
    if i==b:
        alpha=0
    else:
        alpha=i
    for j in range(1, a+1):
        if j==a:
            beta=0
        else:
            beta=j
        H+= J1*(Sx[ij(i, j)].dot(Sx[ij(alpha+1, j)]) + Sy[ij(i, j)].dot(Sy[ij(alpha+1, j)]) + Sz[ij(i, j)].dot(Sz[ij(alpha+1, j)]) +\
                Sx[ij(i, j)].dot(Sx[ij(i, beta+1)]) + Sy[ij(i, j)].dot(Sy[ij(i, beta+1)]) + Sz[ij(i, j)].dot(Sz[ij(i, beta+1)]))
# Next nearest neighbor interaction
for i in range(1, b+1): 
    if i==b: # this two conditions are the periodic boundary condition for each row
        alpha=0
    elif i==b-1:
        alpha=-1
    else:
        alpha=i
    for j in range(1, a+1):
        if j==a: # this two conditions are the periodic boundary condition for each coloumn
            beta=0
        elif j==a-1:
            beta=-1
        else:
            beta=j
        H+= J2*(Sx[ij(i, j)].dot(Sx[ij(alpha+2, j)]) + Sy[ij(i, j)].dot(Sy[ij(alpha+2, j)]) + Sz[ij(i, j)].dot(Sz[ij(alpha+2, j)]) +\
                Sx[ij(i, j)].dot(Sx[ij(i, beta+2)]) + Sy[ij(i, j)].dot(Sy[ij(i, beta+2)]) + Sz[ij(i, j)].dot(Sz[ij(i, beta+2)]))
# projecting hamiltonian in subspace with S_{z, tot}=0
Sz_tot=sp.csr_matrix((2**L, 2**L))
for i in range(1, L+1):
    Sz_tot += Sz[i]
Sz_tot_diag=Sz_tot.diagonal()
F=(Sz_tot_diag==0)
H_p=H[:, F][F, :]

# Diagonalizing the hamiltonian
E, U=sp.linalg.eigs(H_p, k=20, which='SR')
S=E.argsort()
E=np.real(E[S])
U=U[:, S]
psi_p=U[:, 0] # ground state wave-function in projected hilbert space
print('Energy eigenvalues are:')
print(E)

# Plotting energy eigenvalues
plt.figure()
plt.bar(np.arange(1, 21), E)
plt.xlabel('number of state')
plt.ylabel('energy eigenvalue')
labels=[f'{i}' for i in range(1, 21)]
plt.xticks(np.arange(1, 21), labels)
plt.show()
# Computing spin-spin correlation function

# Reconstructing wave-function in the original hilbert space to evaluate the expectation values
psi=np.zeros(2**L)
psi[F]=psi_p
psi=sp.csr_matrix(psi.reshape(-1, 1))

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