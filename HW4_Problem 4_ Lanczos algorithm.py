#%matplotlib qt
import numpy as np
from scipy.linalg import eigh_tridiagonal
import time

# Part a)
N=2000
A=np.random.randn(N, N)
M=(A+A.T)/2

# Obtaining the ground state with numpy's eigh method
def Eigh(H, k=1): # k: number of desired eigenvalues and eigenvectors
    E, W= np.linalg.eigh(H)
    Ek_eigh = E[:k] # ground state eigenvalue
    vk_eigh = W[:, 0:k] # ground state eigenvector
    return (Ek_eigh, vk_eigh)
t1_Eigh=time.time()
Eigh_eigenvalues, Eigh_eigenvectors = Eigh(M, 3)
t2_Eigh=time.time()
time_Eigh= t2_Eigh - t1_Eigh
print("Numpy's eigh method")
print('3 lowest eigenvalues:')
print(Eigh_eigenvalues)
print('Eigenvectors corresponding to 3 lowest eigenvalues:')
print(Eigh_eigenvectors)
print(f"Run time of the numpy's eigh method is {time_Eigh} s.")
print()

# Using Lanczos algorithm
def Lanczos(H, k=1, m=60): 
    '''
    k: number of desired eigenvalues and eigenvectors
    m: number of iterations
    '''
    n=H.shape[0]
    v0= np.random.randn(n, 1)
    v0= v0/(np.linalg.norm(v0))
    a_values=[]
    b_values=[]
    V=np.zeros((n, m))
    c0= np.dot(H, v0)
    a0= np.dot(v0.T, c0)
    a_values.append(a0[0, 0])
    V[:, 0]= v0.reshape(-1)
    #
    w = c0 - a_values[0] * V[:, 0].reshape(-1, 1)
    b = np.linalg.norm(w)
    v = w / b
    c = np.dot(H, v)
    a = np.dot(v.T, c)
    a_values.append(a[0, 0])
    b_values.append(b)
    V[:, 1] = v.reshape(-1)
    #
    for i in range(2, m):
        # b_values has one value less than a_values
        w = c - a_values[i-1] * V[:, i-1].reshape(-1, 1) - b_values[i-2] * V[:, i-2].reshape(-1, 1)
        b= np.linalg.norm(w)
        v= w / b
        c= np.dot(H, v)
        a= np.dot(v.T, c)
        a_values.append(a[0, 0])
        b_values.append(b)
        V[:, i] = v.reshape(-1)
    T= np.diag(a_values) + np.diag(b_values, 1) + np.diag(b_values, -1)
    E, W= eigh_tridiagonal(a_values, b_values)
    Ek_L = E[:k]
    U=np.dot(V, W)
    vk_L = U[:, 0:k]
    return (Ek_L, vk_L)
t1_Lanczos=time.time()
Lanczos_eigenvalues, Lanczos_eigenvectors =Lanczos(M, 3, m=80)
t2_Lanczos=time.time()
time_Lanczos=t2_Lanczos-t1_Lanczos
print('Lanczos algorithm')
print('3 lowest eigenvalues:')
print(Lanczos_eigenvalues)
print('Eigenvectors corresponding to 3 lowest eigenvalues:')
print(Lanczos_eigenvectors)
print(f"Run time of the Lanczos algorithm is {time_Lanczos} s.")
print()

# Check that eigenvectors are close
for i in range(3):
    print(f'|<v{i}_Eigh|v{i}_Lanczos>| = {np.dot(Eigh_eigenvectors[:, i], Lanczos_eigenvectors[:, i])}')
print()
#%%
# part b)
n_b, m_b = 50, 100
A1 = np.random.randn(n_b, n_b)
A2 = np.random.randn(n_b, n_b)
B1 = np.random.randn(m_b, m_b)
B2 = np.random.randn(m_b, m_b)
t1_b=time.time()
M0 = 2*np.kron(A1, B1) + 3*np.kron(A2, B2)
M = M0 + M0.T # M is hemitian
# Numpy's eigh method
t_mid=time.time()
M_eigenvalues_E, M_eigenvectors_E = Eigh(M, k=5)
t2_bE=time.time()
time_bE = t2_bE - t1_b
print('Eigh method for part c, tensor product structure')
print('5 lowest eigenvalues:')
print(M_eigenvalues_E)
print('Eigenvectors corresponding to 5 lowest eigenvalues:')
print(M_eigenvectors_E)
print(f"Run time of the numpy's Eigh is {time_bE} s.")
print()

# Lanczos algorithm
M_eigenvalues, M_eigenvectors = Lanczos(M, k=5, m=100)
t2_bL=time.time()
time_bL = (t2_bL - t1_b) - (t2_bE - t_mid)
print('Lanczos algorithm for part c, tensor product structure')
print('5 lowest eigenvalues:')
print(M_eigenvalues)
print('Eigenvectors corresponding to 5 lowest eigenvalues:')
print(M_eigenvectors)
print(f"Run time of the Lanczos algorithm is {time_bL} s.")
print()

# Check that eigenvectors are close
for i in range(5):
    print(f'|<v{i}_Eigh|v{i}_Lanczos>| = {np.dot(M_eigenvectors_E[:, i], M_eigenvectors[:, i])}')
print()

# part c
n_c = n_b * m_b
def compute_W(g, A, B, C):
    W=np.zeros((m_b, n_b))
    for i in range(len(g)):
        W = W + g[i] * np.dot(B[i], np.dot(C, A[i].T))
    return W
g = [2, 3, 2, 3]
A = [A1, A2, A1.T, A2.T]
B = [B1, B2, B1.T, B2.T]

def modified_Lanczos(g, A, B, k=1, m=60):
    '''
    k: number of desired eigenvalues and eigenvectors
    m: number of iterations
    '''
    C0 = np.random.rand(n_c).reshape(m_b, n_b)
    C0 = C0 / (np.linalg.norm(C0))
    a_values=[]
    b_values=[]
    V=np.zeros((n_c, m))
    C0_r = C0.reshape(n_c)
    W = compute_W(g.copy(), A.copy(), B.copy(), C0.copy())
    W_r = W.reshape(n_c)
    a0 = np.dot(C0_r, W_r)
    a_values.append(a0)
    V[:, 0] = C0_r
    #
    G_r = W_r - a0 * C0_r
    b = np.linalg.norm(G_r)
    C_r = G_r / b
    W = compute_W(g.copy(), A.copy(), B.copy(), C_r.copy().reshape(m_b, n_b))
    W_r = W.reshape(n_c)
    a = np.dot(C_r, W_r)
    a_values.append(a)
    b_values.append(b)
    V[:, 1] = C_r
    for j in range(2, m):
        G_r = W_r.reshape(-1, 1) - a_values[j-1] * V[:, j-1].reshape(-1, 1) - b_values[j-2] * V[:, j-2].reshape(-1, 1)
        b = np.linalg.norm(G_r)
        C_r = G_r / b
        W = compute_W(g.copy(), A.copy(), B.copy(), C_r.copy().reshape(m_b, n_b))
        W_r = W.reshape(n_c, 1)
        a = np.dot(C_r.T, W_r)
        a_values.append(a[0, 0])
        b_values.append(b)
        V[:, j] = C_r.reshape(-1)
    T= np.diag(a_values) + np.diag(b_values, 1) + np.diag(b_values, -1)
    E, W_T= eigh_tridiagonal(a_values, b_values)
    Ek_L = E[:k]
    U=np.dot(V, W_T)
    vk_L = U[:, 0:k]
    return (Ek_L, vk_L)
#
t1 = time.time()
M_eigenvalues_new, M_eigenvectors_new = modified_Lanczos(g, A, B, k=5, m=100)
t2 = time.time()
time_mL = t2 - t1
print("Modified Lanczos algorithm")
print('5 lowest eigenvalues:')
print(M_eigenvalues_new)
print('Eigenvectors corresponding to 5 lowest eigenvalues:')
print(M_eigenvectors_new)
print(f"Run time of the modified Lanczos algorithm is {time_mL} s.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    