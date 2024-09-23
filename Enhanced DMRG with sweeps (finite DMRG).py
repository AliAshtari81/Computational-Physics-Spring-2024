# This program consists of these functions to address spin chains (Heisenberg / 1D Quantum Ising):
# Lanczos : a function to engage Lanczos algorithm to diagonalize hermitian matrices
# Modified Lanczos : More efficient (both in time and in RAM usage) Lanczos algorithm for linear combination of tensor product structures
# DMRG : A thorough function for infinite DMRG and also finite DMRG (using sweeps for enhancing precision) algorithms, for both "Heisenberg chain" and also "1D Quantum Ising chain"

#%matplotlib qt
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.linalg import eigh_tridiagonal
import time


def Lanczos(H, k, m):
    n = H.shape[0]
    V = np.zeros((n, m))
    v0 = np.random.randn(n, 1)
    v0 = v0 / (np.linalg.norm(v0))
    a_values=[]
    b_values=[]
    c0 = np.dot(H, v0)
    a0 = np.dot(v0.T, c0)[0, 0]
    a_values.append(a0) # a0
    V[:, 0] = v0.reshape(-1)
    #
    w1 = c0 - a0 * v0
    b1 = np.linalg.norm(w1)
    v1 = w1 / b1
    V[:, 1] = v1.reshape(-1)
    b_values.append(b1) # b1
    c = np.dot(H, v1)
    a = np.dot(v1.T, c)[0, 0] # a1
    a_values.append(a)
    b = b1
    for i in range(2, m):
        w = c - a * (V[:, i-1].reshape(n, 1)) - b * (V[:, i-2].reshape(n, 1))
        if i%5==0:
            for j in range(i):
                w = w - (np.dot(V[:, j].reshape(1, n), w) * V[:, j]).reshape(n, 1)
        b = np.linalg.norm(w) # b2
        v = w / b # v2
        c = np.dot(H, v)
        a = np.dot(v.T, c)[0, 0] # a2
        b_values.append(b) # b2
        a_values.append(a) # a2
        V[:, i] = v.reshape(-1)
    E_T, W_T = eigh_tridiagonal(a_values, b_values)
    E = E_T[:k]
    U = np.dot(V, W_T)
    U = U[:, :k]
    return (E, U)

def compute_W(g, A, B, C):
    n_A = A[0].shape[0]
    n_B = B[0].shape[0]
    W = np.zeros((n_A, n_B))
    for i in range(len(g)):
        W = W + g[i] * np.dot(A[i], np.dot(C, B[i].T))
    return W

def Modified_Lanczos(g, A, B, k, m):
    '''
    Efficient Lanczos algorithm without explicitely computing tensor products
    '''
    n_A = A[0].shape[0]
    n_B = B[0].shape[0]
    n_tot = n_A * n_B
    C0 = np.random.randn(n_A, n_B)
    a_values = []
    b_values = []
    V = np.zeros((n_tot, m))
    C0_r = C0.reshape(n_tot, 1)
    C0_r = C0_r / (np.linalg.norm(C0_r))
    #
    W0 = compute_W(g, A, B, C0_r.reshape(n_A, n_B))
    W0_r = W0.reshape(n_tot, 1)
    a0 = np.dot(C0_r.T, W0_r)[0, 0] # a0
    a_values.append(a0)
    V[:, 0] = C0_r.reshape(-1) # v0
    #
    W_r = W0_r - a0 * C0_r
    b = np.linalg.norm(W_r) # b1
    C_r = W_r / b
    W = compute_W(g, A, B, C_r.reshape(n_A, n_B))
    W_r = W.reshape(n_tot, 1)
    a = np.dot(C_r.T, W_r)[0, 0] #a1
    b_values.append(b)
    a_values.append(a)
    V[:, 1] = C_r.reshape(-1) # v1
    #
    for i in range(2, m):
        W_r = W_r - a * V[:, i-1].reshape(n_tot, 1) - b * V[:, i-2].reshape(n_tot, 1)
        if i%5==0:
            for j in range(i):
                W_r = W_r - np.dot(V[:, j].reshape(1, n_tot), W_r) * V[:, j].reshape(n_tot, 1)
        b = np.linalg.norm(W_r) #b2
        C_r = W_r / b # v2
        W = compute_W(g, A, B, C_r.reshape(n_A, n_B))
        W_r = W.reshape(n_tot, 1)
        a = np.dot(C_r.T, W_r)[0, 0] # a1
        b_values.append(b)
        a_values.append(a)
        V[:, i] = C_r.reshape(-1)
    T= np.diag(a_values) + np.diag(b_values, 1) + np.diag(b_values, -1)
    E_T, W_T = np.linalg.eigh(T)
    E = E_T[:k]
    U = np.dot(V, W_T)
    U = U[:, :k]
    return (E, U)

hbar = 1
J = 0.5
h = 1.0
def Id(n):
    '''
    Identity operator for n sites
    '''
    if n==0:
        return 1.
    else:
        return np.identity(2**n)
# Truncation
def truncation_matrix(M, n):
    w, u = np.linalg.eigh(M)
    T = u[:, -n:]
    return T
    
def DMRG(L, bd, J, h, Model, sweeps = 0):
    '''
    L : Length
    bd : Bond Dimesion
    Model : 1D Ising / 1D Heisenberg chain
    '''
    sigma_x = np.array([[0., 1.],[1., 0.]])
    sigma_y = np.array([[0., -1.j],[1.j, 0.]])
    sigma_z = np.array([[1., 0.],[0., -1.]])
    I = np.identity(2)
    Id_bd = np.identity(2*bd) # (2*bd) by (2*bd) identity matrix
    Lanczos_iterations = 100
    # Constructing the exact operators for p+1 sites
    # Spin operators
    Sx_L, Sy_L, Sz_L, Sx_R, Sy_R, Sz_R = {}, {}, {}, {}, {}, {} # all of spins at once
    Sx_L_list, Sy_L_list, Sz_L_list, Sx_R_list, Sy_R_list, Sz_R_list = {}, {}, {}, {}, {}, {} # saving spins for using in sweeps
    H_L_list, H_R_list = {}, {} # saving Hamiltonians for using in sweeps
    #
    p = int(np.log2(bd)) # number of particles for exact solution
    Id_L = Id(p+1)
    Id_R = Id(p+1)
    for i in range(1, p+2):
        Sx_L[i] = np.kron(Id(i-1), np.kron((hbar/2)*sigma_x, Id(p+1-i)))
        Sx_L_list[i] = Sx_L[i].copy()
        Sz_L[i] = np.kron(Id(i-1), np.kron((hbar/2)*sigma_z, Id(p+1-i)))
        Sz_L_list[i] = Sz_L[i].copy()
        if Model == 'Heisenberg':
            Sy_L[i] = np.kron(Id(i-1), np.kron((hbar/2)*sigma_y, Id(p+1-i)))
            Sy_L_list[i] = Sy_L[i].copy()
        Sx_R[i] = np.kron(Id(p+1-i), np.kron((hbar/2)*sigma_x, Id(i-1)))
        Sx_R_list[i] = Sx_R[i].copy()
        Sz_R[i] = np.kron(Id(p+1-i), np.kron((hbar/2)*sigma_z, Id(i-1)))
        Sz_R_list[i] = Sz_R[i].copy()
        if Model == 'Heisenberg':
            Sy_R[i] = np.kron(Id(p+1-i), np.kron((hbar/2)*sigma_y, Id(i-1)))
            Sy_R_list[i] = Sy_R[i].copy()
    # Hamiltonian operators
    H_L = np.zeros((2**(p+1), 2**(p+1)))
    H_R = H_L.copy()
    if Model == 'Ising':
        for i in range(1, p+1):
            H_L = H_L - J * np.dot(Sx_L[i], Sx_L[i+1]) - h * Sz_L[i]
            H_R = H_R - J * np.dot(Sx_R[i+1], Sx_R[i]) - h * Sz_R[i]
        H_L = H_L - h * Sz_L[p+1]
        H_R = H_R - h * Sz_R[p+1]
    elif Model == 'Heisenberg':
        for i in range(1, p+1):
            H_L = H_L - J * (np.dot(Sx_L[i], Sx_L[i+1]) + np.dot(Sy_L[i], Sy_L[i+1]) + np.dot(Sz_L[i], Sz_L[i+1]))
            H_R = H_R - J * (np.dot(Sx_R[i+1], Sx_R[i]) + np.dot(Sy_R[i+1], Sy_R[i]) + np.dot(Sz_R[i+1], Sz_R[i]))
    # Note that direction of labeling is oposite in left and right, so the order of multiplication of spins must be exchanged in the right block
    H_L = (H_L + H_L.T)/2
    H_R = (H_R + H_R.T)/2
    H_L_list[p+1] = H_L
    H_R_list[p+1] = H_R
    # Diagonalizing Super_Block Hamiltonian without making it
    if Model == 'Ising':
        g = [1, 1, -J]
        A = [H_L, Id_L, Sx_L[p+1]]
        B = [Id_R, H_R, Sx_R[p+1]]
    elif Model == 'Heisenberg':
        g = [1, 1, -J, -J, -J]
        A = [H_L, Id_L, Sx_L[p+1], Sy_L[p+1], Sz_L[p+1]]
        B = [Id_R, H_R, Sx_R[p+1], Sy_R[p+1], Sz_R[p+1]]
    E, U = Modified_Lanczos(g, A, B, 1, Lanczos_iterations)
    #
    psi = U.reshape(2**(p+1), 2**(p+1))
    rho_L = np.dot(psi, psi.T)
    rho_L = (rho_L + rho_L.T)/2
    rho_R = np.dot(psi.T, psi)
    rho_R = (rho_R + rho_R.T)/2
    # Trucation
    T_L0 = truncation_matrix(rho_L, bd)
    T_R0 = truncation_matrix(rho_R, bd)
    T_L = T_L0.copy()
    T_R = T_R0.copy()
    # Start DMRG iterations
    for l in range(p+2, L//2 + 1):
        for i in range(1, l):
            # Left Block updating
            Sx_L_tilde = np.dot(T_L.T, np.dot(Sx_L[i], T_L))
            Sx_L[i] = np.kron(Sx_L_tilde, I)
            Sz_L_tilde = np.dot(T_L.T, np.dot(Sz_L[i], T_L))
            Sz_L[i] = np.kron(Sz_L_tilde, I)
            if Model == 'Heisenberg':
                Sy_L_tilde = np.dot(T_L.T, np.dot(Sy_L[i], T_L))
                Sy_L[i] = np.kron(Sy_L_tilde, I)
            # Right Block updating
            Sx_R_tilde = np.dot(T_R.T, np.dot(Sx_R[i], T_R))
            Sx_R[i] = np.kron(I, Sx_R_tilde)
            Sz_R_tilde = np.dot(T_R.T, np.dot(Sz_R[i], T_R))
            Sz_R[i] = np.kron(I, Sz_R_tilde)
            if Model == 'Heisenberg':
                Sy_R_tilde = np.dot(T_R.T, np.dot(Sy_R[i], T_R))
                Sy_R[i] = np.kron(I, Sy_R_tilde)
        # Adding a site to the left block
        Sx_L[l] = np.kron(np.identity(bd), (hbar/2)*sigma_x)
        Sx_L_list[l] = Sx_L[l].copy()
        Sz_L[l] = np.kron(np.identity(bd), (hbar/2)*sigma_z)
        Sz_L_list[l] = Sz_L[l].copy()
        if Model == 'Heisenberg':
            Sy_L[l] = np.kron(np.identity(bd), (hbar/2)*sigma_y)
            Sy_L_list[l] = Sy_L[l].copy()
        # Adding a site to the right block
        Sx_R[l] = np.kron((hbar/2)*sigma_x, np.identity(bd))
        Sx_R_list[l] = Sx_R[l].copy()
        Sz_R[l] = np.kron((hbar/2)*sigma_z, np.identity(bd))
        Sz_R_list[l] = Sz_R[l].copy()
        if Model == 'Heisenberg':
            Sy_R[l] = np.kron((hbar/2)*sigma_y, np.identity(bd))
            Sy_R_list[l] = Sy_R[l].copy()
        # Hamiltonian operators
        H_L_tilde = np.dot(T_L.T, np.dot(H_L, T_L))
        if Model == 'Ising':
            H_L = np.kron(H_L_tilde, I) - J * np.kron(Sx_L_tilde, (hbar/2)*sigma_x) - h * Sz_L[l]
        elif Model == 'Heisenberg':
            H_L = np.kron(H_L_tilde, I) - J * (np.kron(Sx_L_tilde, (hbar/2)*sigma_x) +\
                                               np.kron(Sy_L_tilde, (hbar/2)*sigma_y) +\
                                               np.kron(Sz_L_tilde, (hbar/2)* sigma_z))
        H_L = (H_L + H_L.T)/2
        H_L_list[l] = H_L.copy()
        #
        H_R_tilde = np.dot(T_R.T, np.dot(H_R, T_R))
        if Model == 'Ising':
            H_R = np.kron(I, H_R_tilde) - J * np.kron((hbar/2)*sigma_x, Sx_R_tilde) - h * Sz_R[l]
        elif Model == 'Heisenberg':
            H_R = np.kron(I, H_R_tilde) - J * (np.kron((hbar/2)*sigma_x, Sx_R_tilde)+\
                                               np.kron((hbar/2)*sigma_y, Sy_R_tilde)+\
                                               np.kron((hbar/2)*sigma_z, Sz_R_tilde))
        H_R = (H_R + H_R.T)/2
        H_R_list[l] = H_R.copy()
        # Diagonalizing the super-block hamiltonian without making it
        if Model == 'Ising':
            g = [1, 1, -J]
            A = [H_L, Id_bd, Sx_L[l]]
            B = [Id_bd, H_R, Sx_R[l]]
        elif Model == 'Heisenberg':
            g = [1, 1, -J, -J, -J]
            A = [H_L, Id_bd, Sx_L[l], Sy_L[l], Sz_L[l]]
            B = [Id_bd, H_R, Sx_R[l], Sy_R[l], Sz_R[l]]
        E, U = Modified_Lanczos(g, A, B, 1, Lanczos_iterations)
        #
        psi = U.reshape(2*bd, 2*bd)
        rho_L = np.dot(psi, psi.T)
        rho_L = (rho_L + rho_L.T)/2
        rho_R = np.dot(psi.T, psi)
        rho_R = (rho_R + rho_R.T)/2
        # Trucation
        T_L = truncation_matrix(rho_L, bd)
        T_R = truncation_matrix(rho_R, bd)
        if l==(p+2):
            T_R1 = T_R.copy()
            T_L1 = T_L.copy()
        
    Energy = [E[0]]
    # Starting sweeps
    if sweeps != 0:
        for sweep in range(sweeps):
            # Half sweep to right and back to center
            # To Right
            for l in range(L//2 + 1, L - p-1):
                # Left Block extending and updating
                for i in range(1, l):
                    # Updating spins
                    Sx_L_tilde = np.dot(T_L.T, np.dot(Sx_L[i], T_L))
                    Sx_L[i] = np.kron(Sx_L_tilde, I)
                    Sz_L_tilde = np.dot(T_L.T, np.dot(Sz_L[i], T_L))
                    Sz_L[i] = np.kron(Sz_L_tilde, I)
                    if Model == 'Heisenberg':
                        Sy_L_tilde = np.dot(T_L.T, np.dot(Sy_L[i], T_L))
                        Sy_L[i] = np.kron(Sy_L_tilde, I)
                # Adding a site to the left block
                Sx_L[l] = np.kron(np.identity(bd), (hbar/2)*sigma_x)
                Sx_L_list[l] = Sx_L[l].copy()
                Sz_L[l] = np.kron(np.identity(bd), (hbar/2)*sigma_z)
                Sz_L_list[l] = Sz_L[l].copy()
                if Model == 'Heisenberg':
                    Sy_L[l] = np.kron(np.identity(bd), (hbar/2)*sigma_y)
                    Sy_L_list[l] = Sy_L[l].copy()
                # Hamiltonian operator
                H_L_tilde = np.dot(T_L.T, np.dot(H_L, T_L))
                if Model == 'Ising':
                    H_L = np.kron(H_L_tilde, I) - J * np.kron(Sx_L_tilde, (hbar/2)*sigma_x) - h * Sz_L[l]
                elif Model == 'Heisenberg':
                    H_L = np.kron(H_L_tilde, I) - J * (np.kron(Sx_L_tilde, (hbar/2)*sigma_x) +\
                                                       np.kron(Sy_L_tilde, (hbar/2)*sigma_y) +\
                                                       np.kron(Sz_L_tilde, (hbar/2)* sigma_z))
                H_L = (H_L + H_L.T)/2
                H_L_list[l] = H_L.copy()
                # Right Block shrinking
                # Diagonalizing the super-block hamiltonian without making it
                if Model == 'Ising':
                    g = [1, 1, -J]
                    A = [H_L, Id_bd, Sx_L[l]]
                    B = [Id_bd, H_R_list[L-l], Sx_R_list[L-l]]
                elif Model == 'Heisenberg':
                    g = [1, 1, -J, -J, -J]
                    A = [H_L, Id_bd, Sx_L[l], Sy_L[l], Sz_L[l]]
                    B = [Id_bd, H_R_list[L-l], Sx_R_list[L-l], Sy_R_list[L-l], Sz_R_list[L-l]] # Loading data of right block
                E, U = Modified_Lanczos(g, A, B, 1, Lanczos_iterations)
                # Truncation in left block
                psi = U.reshape(2*bd, 2*bd)
                rho_L = np.dot(psi, psi.T)
                rho_L = (rho_L + rho_L.T)/2
                T_L = truncation_matrix(rho_L, bd)
            # site L-p-1 in left block
            for i in range(1, L-p-1):
                # Updating spins
                Sx_L_tilde = np.dot(T_L.T, np.dot(Sx_L[i], T_L))
                Sx_L[i] = np.kron(Sx_L_tilde, I)
                Sz_L_tilde = np.dot(T_L.T, np.dot(Sz_L[i], T_L))
                Sz_L[i] = np.kron(Sz_L_tilde, I)
                if Model == 'Heisenberg':
                    Sy_L_tilde = np.dot(T_L.T, np.dot(Sy_L[i], T_L))
                    Sy_L[i] = np.kron(Sy_L_tilde, I)
            # Adding a site to the left block
            Sx_L[L-p-1] = np.kron(np.identity(bd), (hbar/2)*sigma_x)
            Sx_L_list[L-p-1] = Sx_L[L-p-1].copy()
            Sz_L[L-p-1] = np.kron(np.identity(bd), (hbar/2)*sigma_z)
            Sz_L_list[L-p-1] = Sz_L[L-p-1].copy()
            if Model == 'Heisenberg':
                Sy_L[L-p-1] = np.kron(np.identity(bd), (hbar/2)*sigma_y)
                Sy_L_list[L-p-1] = Sy_L[L-p-1].copy()
            # Hamiltonian operator
            H_L_tilde = np.dot(T_L.T, np.dot(H_L, T_L))
            if Model == 'Ising':
                H_L = np.kron(H_L_tilde, I) - J * np.kron(Sx_L_tilde, (hbar/2)*sigma_x) - h * Sz_L[L-p-1]
            elif Model == 'Heisenberg':
                H_L = np.kron(H_L_tilde, I) - J * (np.kron(Sx_L_tilde, (hbar/2)*sigma_x) +\
                                                   np.kron(Sy_L_tilde, (hbar/2)*sigma_y) +\
                                                   np.kron(Sz_L_tilde, (hbar/2)* sigma_z))
            H_L = (H_L + H_L.T)/2
            H_L_list[L-p-1] = H_L.copy()
            # Right Block shrinking
            # Diagonalizing the super-block hamiltonian without making it
            if Model == 'Ising':
                g = [1, 1, -J]
                A = [H_L, Id_bd, Sx_L[L-p-1]]
                B = [Id_R, H_R_list[p+1], Sx_R_list[p+1]]
            elif Model == 'Heisenberg':
                g = [1, 1, -J, -J, -J]
                A = [H_L, Id_bd, Sx_L[L-p-1], Sy_L[L-p-1], Sz_L[L-p-1]]
                B = [Id_R, H_R_list[p+1], Sx_R_list[p+1], Sy_R_list[p+1], Sz_R_list[p+1]] # Loading data of right block
            E, U = Modified_Lanczos(g, A, B, 1, Lanczos_iterations)
            # Truncation in left block
            psi = U.reshape(2*bd, 2**(p+1))
            rho_L = np.dot(psi, psi.T)
            rho_L = (rho_L + rho_L.T)/2
            T_L = truncation_matrix(rho_L, bd)   
            # Back to left
            # Diagonalizing the super-block hamiltonian without making it
            if Model == 'Ising':
                g = [1, 1, -J]
                A = [H_L, Id_bd, Sx_L[L-p-1]]
                B = [Id_R, H_R_list[p+1], Sx_R_list[p+1]]
            elif Model == 'Heisenberg':
                g = [1, 1, -J, -J, -J]
                A = [H_L, Id_bd, Sx_L[L-p-1], Sy_L[L-p-1], Sz_L[L-p-1]]
                B = [Id_R, H_R_list[p+1], Sx_R_list[p+1], Sy_R_list[p+1], Sz_R_list[p+1]] 
            E, U = Modified_Lanczos(g, A, B, 1, Lanczos_iterations)
            psi = U.reshape(2*bd, 2**(p+1))
            rho_R = np.dot(psi.T, psi)
            rho_R = (rho_R + rho_R.T)/2
            T_R = truncation_matrix(rho_R, bd)
            #
            for i in range(1, p+2):
                Sx_R[i] = Sx_R_list[i].copy()
                Sz_R[i] = Sz_R_list[i].copy()
                if Model == 'Heisenberg':
                    Sy_R[i] = Sy_R_list[i].copy() 
            H_R = H_R_list[p+1]
            for l in range(p+2, L-p-1):
                for i in range(1, l):
                    # Right Block updating
                    Sx_R_tilde = np.dot(T_R.T, np.dot(Sx_R[i], T_R))
                    Sx_R[i] = np.kron(I, Sx_R_tilde)
                    Sz_R_tilde = np.dot(T_R.T, np.dot(Sz_R[i], T_R))
                    Sz_R[i] = np.kron(I, Sz_R_tilde)
                    if Model == 'Heisenberg':
                        Sy_R_tilde = np.dot(T_R.T, np.dot(Sy_R[i], T_R))
                        Sy_R[i] = np.kron(I, Sy_R_tilde)
                # Adding a site to the right block
                Sx_R[l] = np.kron((hbar/2)*sigma_x, np.identity(bd))
                Sx_R_list[l] = Sx_R[l].copy()
                Sz_R[l] = np.kron((hbar/2)*sigma_z, np.identity(bd))
                Sz_R_list[l] = Sz_R[l].copy()
                if Model == 'Heisenberg':
                    Sy_R[l] = np.kron((hbar/2)*sigma_y, np.identity(bd))
                    Sy_R_list[l] = Sy_R[l].copy()
                # Hamiltonian operator
                H_R_tilde = np.dot(T_R.T, np.dot(H_R, T_R))
                if Model == 'Ising':
                    H_R = np.kron(I, H_R_tilde) - J * np.kron((hbar/2)*sigma_x, Sx_R_tilde) - h * Sz_R[l]
                elif Model == 'Heisenberg':
                    H_R = np.kron(I, H_R_tilde) - J * (np.kron((hbar/2)*sigma_x, Sx_R_tilde)+\
                                                       np.kron((hbar/2)*sigma_y, Sy_R_tilde)+\
                                                       np.kron((hbar/2)*sigma_z, Sz_R_tilde))
                H_R = (H_R + H_R.T)/2
                H_R_list[l] = H_R.copy()
                # Diagonalizing the super-block hamiltonian without making it
                if Model == 'Ising':
                    g = [1, 1, -J]
                    A = [H_L_list[L-l], Id_bd, Sx_L_list[L-l]]
                    B = [Id_bd, H_R, Sx_R[l]]
                elif Model == 'Heisenberg':
                    g = [1, 1, -J, -J, -J]
                    A = [H_L_list[L-l], Id_bd, Sx_L_list[L-l], Sy_L_list[L-l], Sz_L_list[L-l]]
                    B = [Id_bd, H_R, Sx_R[l], Sy_R[l], Sz_R[l]]
                E, U = Modified_Lanczos(g, A, B, 1, Lanczos_iterations)
                #
                psi = U.reshape(2*bd, 2*bd)
                rho_R = np.dot(psi.T, psi)
                rho_R = (rho_R + rho_R.T)/2
                # Trucation
                T_R = truncation_matrix(rho_R, bd)
            # Site L-p-1 in right block
            for i in range(1, L-p-1):
                # Right Block updating
                Sx_R_tilde = np.dot(T_R.T, np.dot(Sx_R[i], T_R))
                Sx_R[i] = np.kron(I, Sx_R_tilde)
                Sz_R_tilde = np.dot(T_R.T, np.dot(Sz_R[i], T_R))
                Sz_R[i] = np.kron(I, Sz_R_tilde)
                if Model == 'Heisenberg':
                    Sy_R_tilde = np.dot(T_R.T, np.dot(Sy_R[i], T_R))
                    Sy_R[i] = np.kron(I, Sy_R_tilde)
            # Adding a site to the right block
            Sx_R[L-p-1] = np.kron((hbar/2)*sigma_x, np.identity(bd))
            Sx_R_list[L-p-1] = Sx_R[L-p-1].copy()
            Sz_R[L-p-1] = np.kron((hbar/2)*sigma_z, np.identity(bd))
            Sz_R_list[L-p-1] = Sz_R[L-p-1].copy()
            if Model == 'Heisenberg':
                Sy_R[L-p-1] = np.kron((hbar/2)*sigma_y, np.identity(bd))
                Sy_R_list[L-p-1] = Sy_R[L-p-1].copy()
            # Hamiltonian operator
            H_R_tilde = np.dot(T_R.T, np.dot(H_R, T_R))
            if Model == 'Ising':
                H_R = np.kron(I, H_R_tilde) - J * np.kron((hbar/2)*sigma_x, Sx_R_tilde) - h * Sz_R[L-p-1]
            elif Model == 'Heisenberg':
                H_R = np.kron(I, H_R_tilde) - J * (np.kron((hbar/2)*sigma_x, Sx_R_tilde)+\
                                                   np.kron((hbar/2)*sigma_y, Sy_R_tilde)+\
                                                   np.kron((hbar/2)*sigma_z, Sz_R_tilde))
            H_R = (H_R + H_R.T)/2
            H_R_list[L-p-1] = H_R.copy()
            # Diagonalizing the super-block hamiltonian without making it
            if Model == 'Ising':
                g = [1, 1, -J]
                A = [H_L_list[p+1], Id_L, Sx_L_list[p+1]]
                B = [Id_bd, H_R, Sx_R[L-p-1]]
            elif Model == 'Heisenberg':
                g = [1, 1, -J, -J, -J]
                A = [H_L_list[p+1], Id_L, Sx_L_list[p+1], Sy_L_list[p+1], Sz_L_list[p+1]]
                B = [Id_bd, H_R, Sx_R[L-p-1], Sy_R[L-p-1], Sz_R[L-p-1]]
            E, U = Modified_Lanczos(g, A, B, 1, Lanczos_iterations)
            #
            psi = U.reshape(2**(p+1), 2*bd)
            rho_R = np.dot(psi.T, psi)
            rho_R = (rho_R + rho_R.T)/2
            # Trucation
            T_R = truncation_matrix(rho_R, bd)
            # Back to right
            # Diagonalizing the super-block hamiltonian without making it
            if Model == 'Ising':
                g = [1, 1, -J]
                A = [H_L_list[p+1], Id_L, Sx_L_list[p+1]]
                B = [Id_bd, H_R, Sx_R[L-p-1]]
            elif Model == 'Heisenberg':
                g = [1, 1, -J, -J, -J]
                A = [H_L_list[p+1], Id_L, Sx_L_list[p+1], Sy_L_list[p+1], Sz_L_list[p+1]]
                B = [Id_bd, H_R, Sx_R[L-p-1], Sy_R[L-p-1], Sz_R[L-p-1]] 
            E, U = Modified_Lanczos(g, A, B, 1, Lanczos_iterations)
            psi = U.reshape(2**(p+1), 2*bd)
            rho_L = np.dot(psi, psi.T)
            rho_L = (rho_L + rho_L.T)/2
            T_L = truncation_matrix(rho_L, bd)
            #
            for i in range(1, p+2):
                Sx_L[i] = Sx_L_list[i].copy()
                Sz_L[i] = Sz_L_list[i].copy()
                if Model == 'Heisenberg':
                    Sy_L[i] = Sy_L_list[i].copy() 
            H_L = H_L_list[p+1]
            for l in range(p+2, L//2 + 1):
                for i in range(1, l):
                    # Left Block updating
                    Sx_L_tilde = np.dot(T_L.T, np.dot(Sx_L[i], T_L))
                    Sx_L[i] = np.kron(Sx_L_tilde, I)
                    Sz_L_tilde = np.dot(T_L.T, np.dot(Sz_L[i], T_L))
                    Sz_L[i] = np.kron(Sz_L_tilde, I)
                    if Model == 'Heisenberg':
                        Sy_L_tilde = np.dot(T_L.T, np.dot(Sy_L[i], T_L))
                        Sy_L[i] = np.kron(Sy_L_tilde, I)
                # Adding a site to the left block
                Sx_L[l] = np.kron(np.identity(bd), (hbar/2)*sigma_x)
                Sx_L_list[l] = Sx_L[l].copy()
                Sz_L[l] = np.kron(np.identity(bd), (hbar/2)*sigma_z)
                Sz_L_list[l] = Sz_L[l].copy()
                if Model == 'Heisenberg':
                    Sy_L[l] = np.kron(np.identity(bd), (hbar/2)*sigma_y)
                    Sy_L_list[l] = Sy_L[l].copy()
                # Hamiltonian operator
                H_L_tilde = np.dot(T_L.T, np.dot(H_L, T_L))
                if Model == 'Ising':
                    H_L = np.kron(H_L_tilde, I) - J * np.kron(Sx_L_tilde, (hbar/2)*sigma_x) - h * Sz_L[l]
                elif Model == 'Heisenberg':
                    H_L = np.kron(H_L_tilde, I) - J * (np.kron(Sx_L_tilde, (hbar/2)*sigma_x) +\
                                                       np.kron(Sy_L_tilde, (hbar/2)*sigma_y) +\
                                                           np.kron(Sz_L_tilde, (hbar/2)* sigma_z))
                    H_L = (H_L + H_L.T)/2
                    H_L_list[l] = H_L.copy()
                # Right Block shrinking
                # Diagonalizing the super-block hamiltonian without making it
                if Model == 'Ising':
                    g = [1, 1, -J]
                    A = [H_L, Id_bd, Sx_L[l]]
                    B = [Id_bd, H_R_list[L-l], Sx_R_list[L-l]]
                elif Model == 'Heisenberg':
                    g = [1, 1, -J, -J, -J]
                    A = [H_L, Id_bd, Sx_L[l], Sy_L[l], Sz_L[l]]
                    B = [Id_bd, H_R_list[L-l], Sx_R_list[L-l], Sy_R_list[L-l], Sz_R_list[L-l]] # Loading data of right block
                E, U = Modified_Lanczos(g, A, B, 1, Lanczos_iterations)
                # Truncation in left block
                psi = U.reshape(2*bd, 2*bd)
                rho_L = np.dot(psi, psi.T)
                rho_L = (rho_L + rho_L.T)/2
                T_L = truncation_matrix(rho_L, bd)
            Energy.append(E[0])
    return Energy
# Debug process
                
            

#def wavefuncton_transformation(Psi, T_L, T_R, D_L, D_R):
#    Psi = np.dot(T_L.T, Psi)
#    Psi_new = np.reshape(Psi, (D_L//2, D_R//2, 2))
#    Psi_new = np.transpose(Psi_new, (0, 2, 1))
#    Psi_new = np.reshape(Psi_new, (D_L, D_R//2))
#    Psi = np.dot(Psi_new, T_R.T)
#    return Psi.reshape(-1, 1)
#T_L = np.random.randn(32, 16)
#T_R = np.random.randn(32, 16)
#Psi = np.random.randn(32, 32)
#D_L, D_R = 32, 32
#print(wavefuncton_transformation(Psi, T_L, T_R, D_L, D_R))

    

# Testin for an Ising chaing with L = 50 sites, bd = 30, J = 1.0 and h = 1.0 with 3 sweeps
t1 = time.time()
Energy = DMRG(50, 30, 1, h, 'Ising', sweeps=3)
print(Energy)
t2 = time.time()
t = t2 - t1
print(f'Run time of the program is {t//60} min, {t%60} s.')
