#%matplotlib qt
#The above line is to show the plots better (in Spyder), and may not work in some computers. So it can be a comment.
import numpy as np
import matplotlib.pyplot as plt
dx=0.01
x=np.arange(-10+dx, 10, dx) #ignoring first and last points
N=len(x)
omega, m, hbar=1, 1, 1
# construcing D2 operator : D2=d^2/dx^2
d1=np.ones(N)
d2=np.ones(N-1)
D2=(1/dx**2)*(np.diag(-2*d1) + np.diag(d2, 1) + np.diag(d2, -1))
# introducing potential functions
def V1(x): # simple harmonic oscillator
    return 0.5*m*(omega**2)*x**2
def V2(x): # infinite square well
    if abs(x)<3:
        return 0
    else:
        return 1e6
def V3(x): # cosine potential function
    return 1-np.cos(x/10)
def V4(x): # linear distance potential
    return abs(x)
# Construcing and diagonalizing hamiltonian for each of the potentials, and plotting the result for wavefunctions and eigenenergies

### Simple harmonic oscilator
H1=((-hbar**2)/(2*m))*D2 + np.diag(np.vectorize(V1)(x))
E1, psi1=np.linalg.eigh(H1)
# plotting the potential function
plt.figure()
plt.plot(x, V1(x))
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Potential function for simple harmonic oscillator')
plt.grid()
plt.show()
# plotting the wavefunctions
plt.figure()
plt.plot(x, psi1[:, 0], label='ground state')
plt.plot(x, psi1[:, 1], label='1st excited state')
plt.xlabel('x')
plt.ylabel('$\psi$')
plt.title('ground state & first excited state wavefunctions for simple harmonic oscillator')
plt.legend()
plt.grid()
plt.show()
# plotting the probability distribution function
plt.figure()
plt.plot(x, (psi1[:, 0])**2, label='ground state')
plt.plot(x, (psi1[:, 1])**2, label='first excited state')
plt.xlabel('x')
plt.ylabel('$P=|\psi|^2$')
plt.title('ground state & first excited state probability distribution for simple harmonic oscillator')
plt.legend()
plt.grid()
plt.show()
# plotting energy eigenvalues
plt.figure()
Es1=[]
for i in range(6):
    Es1.append(fr'$E_{i} = {E1[i]:.4f}$')
plt.bar(Es1, E1[:6])
plt.xlabel('eigenenergies')
plt.ylabel('values')
plt.title('Lowest 6 eigenenergies of simple harmonic oscilator')
plt.show()

### Infinite square well
H2=((-hbar**2)/(2*m))*D2 + np.diag(np.vectorize(V2)(x))
E2, psi2=np.linalg.eigh(H2)
# plotting the potential function
plt.figure()
plt.plot(x, np.vectorize(V2)(x))
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Potential function for infinite square well')
plt.grid()
plt.show()
# plotting the wavefunctions
plt.figure()
plt.plot(x, psi2[:, 0], label='ground state')
plt.plot(x, psi2[:, 1], label='1st excited state')
plt.xlabel('x')
plt.ylabel('$\psi$')
plt.title('ground state & first excited state wavefunctions for infinite square well')
plt.legend()
plt.grid()
plt.show()
# plotting the probability distribution function
plt.figure()
plt.plot(x, (psi2[:, 0])**2, label='ground state')
plt.plot(x, (psi2[:, 1])**2, label='first excited state')
plt.xlabel('x')
plt.ylabel('$P=|\psi|^2$')
plt.title('ground state & first excited state probability distribution for infinite square well')
plt.legend()
plt.grid()
plt.show()
# plotting energy eigenvalues
plt.figure()
Es2=[]
for i in range(6):
    Es2.append(fr'$E_{i} = {E2[i]:.4f}$')
plt.bar(Es2, E2[:6])
plt.xlabel('eigenenergies')
plt.ylabel('values')
plt.title('Lowest 6 eigenenergies of infinite square well')
plt.show()

### Cosine potential function
H3=((-hbar**2)/(2*m))*D2 + np.diag(np.vectorize(V3)(x))
E3, psi3=np.linalg.eigh(H3)
# plotting the potential function
plt.figure()
plt.plot(x, np.vectorize(V3)(x))
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Potential function for cosine potential function')
plt.grid()
plt.show()
# plotting the wavefunctions
plt.figure()
plt.plot(x, psi3[:, 0], label='ground state')
plt.plot(x, psi3[:, 1], label='1st excited state')
plt.xlabel('x')
plt.ylabel('$\psi$')
plt.title('ground state & first excited state wavefunctions for cosine potential function')
plt.legend()
plt.grid()
plt.show()
# plotting the probability distribution function
plt.figure()
plt.plot(x, (psi3[:, 0])**2, label='ground state')
plt.plot(x, (psi3[:, 1])**2, label='first excited state')
plt.xlabel('x')
plt.ylabel('$P=|\psi|^2$')
plt.title('ground state & first excited state probability distribution for cosine potential function')
plt.legend()
plt.grid()
plt.show()
# plotting energy eigenvalues
plt.figure()
Es3=[]
for i in range(6):
    Es3.append(fr'$E_{i} = {E3[i]:.4f}$')
plt.bar(Es3, E3[:6])
plt.xlabel('eigenenergies')
plt.ylabel('values')
plt.title('Lowest 6 eigenenergies of cosine potential function')
plt.show()

### Linear distance potential
H4=((-hbar**2)/(2*m))*D2 + np.diag(np.vectorize(V4)(x))
E4, psi4=np.linalg.eigh(H4)
# plotting the potential function
plt.figure()
plt.plot(x, np.vectorize(V4)(x))
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Potential function for linear distance potential')
plt.grid()
plt.show()
# plotting the wavefunctions
plt.figure()
plt.plot(x, psi4[:, 0], label='ground state')
plt.plot(x, psi4[:, 1], label='1st excited state')
plt.xlabel('x')
plt.ylabel('$\psi$')
plt.title('ground state & first excited state wavefunctions for linear distance potential')
plt.legend()
plt.grid()
plt.show()
# plotting the probability distribution function
plt.figure()
plt.plot(x, (psi4[:, 0])**2, label='ground state')
plt.plot(x, (psi4[:, 1])**2, label='first excited state')
plt.xlabel('x')
plt.ylabel('$P=|\psi|^2$')
plt.title('ground state & first excited state probability distribution for linear distance potential')
plt.legend()
plt.grid()
plt.show()
# plotting energy eigenvalues
plt.figure()
Es4=[]
for i in range(6):
    Es4.append(fr'$E_{i} = {E4[i]:.4f}$')
plt.bar(Es4, E4[:6])
plt.xlabel('eigenenergies')
plt.ylabel('values')
plt.title('Lowest 6 eigenenergies of linear distance potential')
plt.show()

# Some sanity checks in linear algebra
print('Results of the sanity checks:')
M0=10*(np.random.randn(3, 3) + 1j*np.random.rand(3, 3)) # constructing a random 3*3 matrix
M1=(M0 + M0.T.conjugate())/2 #constructing a hermitian matrix
Md, U=np.linalg.eigh(M1)
M1_new=np.matmul(U, np.matmul(np.diag(Md), U.T.conjugate()))
print('M1 = M1_new ?   ', np.allclose(M1, M1_new)) # checked that if M1 & M1_new are close enough to be considered equal(They are equal in theory, but there is an error in numerical work)
v1, v2, v3 = U[:, 0], U[:, 1], U[:, 2] # |v1>, |v2>, |v3>
# checking orthonormality
print('overlap_11 = ', np.vdot(v1, v1))
print('overlap_12 = ', np.vdot(v1, v2))
print('overlap_13 = ', np.vdot(v1, v3))
print('overlap_21 = ', np.vdot(v2, v1))
print('overlap_22 = ', np.vdot(v2, v2))
print('overlap_23 = ', np.vdot(v2, v3))
print('overlap_31 = ', np.vdot(v3, v1))
print('overlap_32 = ', np.vdot(v3, v2))
print('overlap_33 = ', np.vdot(v3, v3))
# checking U to be unitary
print('U(U^dag) = ', np.matmul(U, U.T.conjugate()))
print('U(U^dag) = 1?   ', np.allclose(np.identity(3), np.matmul(U, U.T.conjugate())))
print('(U^dag)U = ', np.matmul(U.T.conjugate(), U))
print('(U^dag)U = 1?   ', np.allclose(np.identity(3), np.matmul(U.T.conjugate(), U)))
# checking resolution of identity
outer_products= np.outer(v1, v1.conjugate()) + np.outer(v2, v2.conjugate()) + np.outer(v3, v3.conjugate())
print('\sum_{j} |v_j> < v_j| = ', outer_products)
print('Is it equal to identity?   ', np.allclose(np.identity(3), outer_products))