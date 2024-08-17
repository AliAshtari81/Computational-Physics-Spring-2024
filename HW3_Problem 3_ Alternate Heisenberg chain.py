#%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time # for evaluating the run time of the program
t_initial=time.time()
L=12
J1=1.
J2=0.
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
# Constructing and diagonalizing the initial hamiltonian
H0=sp.csr_matrix((2**L, 2**L))
for i in range(1, L, 2):
    H0+= J1*(Sx[i].dot(Sx[i+1]) + Sy[i].dot(Sy[i+1]) + Sz[i].dot(Sz[i+1]))
E0, U0= sp.linalg.eigs(H0, k=5, which='SR')
S=E0.argsort()
E0=np.real(E0[S])
U0=U0[:, S]
# Ground state energy and wave-function
print(f'Ground state energy is {E0[0]}')
psi_0=U0[:, 0]
pdf=(psi_0.conj())*(psi_0)
plt.plot(np.linspace(1, L, 2**L), pdf)
plt.grid()
plt.xlabel('x')
plt.ylabel('$| \psi |^2$')
# Constructing new hamiltonian
H=sp.csr_matrix((2**L, 2**L))
for i in range(1, L):
    H+= J1*(Sx[i].dot(Sx[i+1]) + Sy[i].dot(Sy[i+1]) + Sz[i].dot(Sz[i+1]))
if bnd==1: # applying periodic boundary condition
    H+=J1*(Sx[L].dot(Sx[1]) + Sy[L].dot(Sy[1]) + Sz[L].dot(Sz[1]))
# calculating time evolution
dt=0.05
T=10
t0=0
N=int(T/dt)
x=np.linspace(1, 16, 2**L)
Psi={t0: psi_0}
psi=psi_0
U=sp.linalg.expm((-1j*dt/hbar)*H)
#U=sp.identity(2**L, format='csr') - ((1j*dt/hbar)*H)
for l in range(1, N+1):
    #psi=(sp.identity(2**L, format='csr')-((1j*l*dt/hbar)*H)).dot(psi)
    #psi/=(np.linalg.norm(psi))
    psi=U.dot(psi)
    #psi=sp.linalg.expm_multiply((-1j*l*dt/hbar)*H, psi_0)
    t0+=dt
    Psi[t0]=psi

# Calculating the desired expectation values
#<S_i(t).S_j(t)>
plt.figure()
i=1 # fixing i because of translational symmetry
for j in range(1, L+1):
    o=Sx[i].dot(Sx[j]) + Sy[i].dot(Sy[j]) + Sz[i].dot(Sz[j])
    Corr={}
    for t, psi_t in Psi.items():
        Corr[t] = (psi_t.T.conj()).dot(o.dot(psi_t))
    plt.plot(Corr.keys(), Corr.values(), label=f'distance = j - i = {j-i}')
plt.xlabel('time')
plt.ylabel('spin-spin correlation function')
plt.title('$<S_i(t).S_j(t)>$')
plt.grid()
plt.legend()
plt.show()
 
#<S_i(t).S_j(0)>
plt.figure()
i=1 # fixing i because of translational symmetry
for j in range(1, L+1):
    o=Sx[i].dot(Sx[j]) + Sy[i].dot(Sy[j]) + Sz[i].dot(Sz[j])
    Corr={}
    for t, psi_t in Psi.items():
        Corr[t] = (psi_t.T.conj()).dot(o.dot(psi_0))
    plt.plot(Corr.keys(), Corr.values(), label=f'distance = j - i = {j-i}')
plt.xlabel('time')
plt.ylabel('spin-spin correlation function')
plt.grid()
plt.legend()
plt.title('$<S_i(t).S_j(0)>$')
plt.show()

# Calculating entanglement entropy of left half with right half
EE={}
for t, psi_t in Psi.items():
    psi_t_tilda=psi_t.reshape(2**(int(L/2)), 2**(int(L/2)))
    rho_L=(psi_t_tilda.T).dot(psi_t_tilda.conj())
    p, v = np.linalg.eigh(rho_L)
    p[abs(p)<1e-12]=1e-12
    EE[t]=-p.dot(np.log(p))
print(EE)
# Plotting entanglement entropy
plt.figure()
plt.plot(EE.keys(), EE.values())
plt.xlabel('time')
plt.ylabel('entanglement entropy')
plt.grid()
plt.show()
t_final=time.time()
print(f'Run time of the program is {t_final - t_initial} s')
    










