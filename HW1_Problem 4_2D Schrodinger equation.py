#%matplotlib qt
#The above line is to show the plots better (in Spyder), and may not work in some computers. So it can be a comment.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
dx=0.2
x=np.arange(-4+dx, 4, dx) #discarding two endpoints
y=np.arange(-4+dx, 4, dx) #discarding two endpoints
X, Y=np.meshgrid(x, y) # forming the grid
N=len(x)
omega, m, hbar= 1., 1., 1.
# constructing second order differntial operator D2=d^2/dx^2
d1=np.ones(N)
d2=np.ones(N-1)
D2=(np.diag(-2*d1) + np.diag(d2, -1) + np.diag(d2, 1))/dx**2
D2_sp=sp.csr_matrix(D2) # converting to sparse matrix
I_sp=sp.identity(N, format='csr') # sparse identity matrix
# all of the above operators are in fact for x_space, but because we construction the grid so that x_space and y_space are similar, all of them can be used for y_space too.

# constructing total second order differentiaition operator (Laplacian)
D_tot=sp.kron(D2_sp, I_sp) + sp.kron(I_sp, D2_sp)
def f(X, Y): # potential of 2D simple harmonic oscillator
    return 0.5*m*(omega**2)*(X**2 + Y**2)
V=f(X, Y)
V_sp=sp.diags(V.reshape(N**2), 0, format='csr')
# constructing the hamiltonian
H=((-hbar**2)/(2*m))*D_tot + V_sp
E, psi=sp.linalg.eigsh(H, k=10, which='SM') # diagonalizing the hamiltonain
# plotting the potential function
fig1=plt.figure()
ax1=fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, V, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('V(X, Y)')
ax1.set_title('Potential function')
plt.show()
# plotting ground state probability distribution function
fig2=plt.figure()
ax2=fig2.add_subplot(111, projection='3d')
psi_0=(psi[:, 0]).reshape((N, N)) # ground state wavefunction
ax2.plot_surface(X, Y, psi_0**2, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('$P =| \psi (X, Y)|^2$')
ax2.set_title('Probability distribution function for ground state')
plt.show()
# plotting first excited state probability distribution function
fig3=plt.figure()
ax3=fig3.add_subplot(111, projection='3d')
psi_1=(psi[:, 1].reshape((N, N))) # 1st excited state wavefunction
ax3.plot_surface(X, Y, psi_1**2, cmap='viridis')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('$P =| \psi (X, Y)|^2$')
ax3.set_title('Probability distribution function for first excited state')
plt.show()
# plotting energy eigenvalues
Es=[]
for i in range(10):
    Es.append(fr'$E_{i} = {E[i]:.4f}$')
plt.figure()
plt.bar(Es, E)
plt.xlabel('eigenenergies')
plt.ylabel('values')
plt.title('10 lowest eigenenergies')
plt.show()