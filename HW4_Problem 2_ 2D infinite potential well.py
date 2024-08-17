#%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from mpl_toolkits.mplot3d import Axes3D
import time
t_initial=time.time()

hbar, m= 1., 1.
# Creating the grid
N=100
dx=20/(N+1) # dy=dx
x=np.linspace(-10, 10, N+2)[1:-1]
y=np.linspace(-10, 10, N+2)[1:-1]
X, Y=np.meshgrid(x, y)
# Constructing the operators
d1=np.ones(N)
d2=np.ones(N-1)
D=(-2*np.diag(d1) + np.diag(d2, 1) + np.diag(d2, -1))/(dx**2)
D=sp.csr_matrix(D) # Dy=Dx=D
Id=sp.identity(N, format='csr') # Id_x = Id_y = Id
D_tot= sp.kron(D, Id) + sp.kron(Id, D)
# Constructing and plotting potential function
def f(x, y):
    if (abs(x)<=5) and (abs(y)<=5):
        return 0
    else:
        return 1e8
V=np.vectorize(f)(X, Y)
fig1=plt.figure()
ax1=fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, V, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('V(X, Y)')
ax1.set_title('Potential function')

# Construcing and diagonalizing the hamiltonian
V_sp=sp.diags(V.reshape(N**2), 0, format='csr')
H=((-hbar**2)/(2*m))*D_tot + V_sp
E, U= sp.linalg.eigs(H, k=10, which='SR')
# Sorting eigenvalues accending
S=E.argsort()
E=np.real(E[S])
U=U[:, S]
print('Energy eigenvalues are: ')
print(E)
# Plotting energy eigenvalues
plt.figure()
E_labels=[]
for i in range(10):
    E_labels.append(fr'$E_{i+1} = {E[i]:.4f}$')
plt.bar(E_labels, E)
plt.xlabel('number of state')
plt.ylabel('energy eigenvalue')
# Plotting ground state wave-function
psi_0 = (U[:, 0]).reshape(N, N) # ground state wave-function
fig2=plt.figure()
ax2=fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, psi_0, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('$\psi_0 (X, Y)$')
ax2.set_title('Ground-state wave function')
# Plotting first excited state wave-function
psi_1=(U[:, 1]).reshape((N, N)) # first excited state wave-function
fig3=plt.figure()
ax3=fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, Y, psi_1, cmap='viridis')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('$\psi_1 (X, Y)$')
ax3.set_title('First excited state wave function')




t_final=time.time()
print(f'Run time of the program is {t_final - t_initial} s')












