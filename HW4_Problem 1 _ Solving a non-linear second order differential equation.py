#%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
t_initial=time.time()
#
(m, k, a) = (5, 2, 4) # setting the constant values
(x0, Dx0) = (1, 0)
# m*(d^2x/dt^2) + k*x = a*x^3/6

# Using odeint to solve the equation
t1_o=time.time()
def system(x, t):
    x1, x2 = x
    dx1dt = x2
    dx2dt = ((a*x1**3)/6 - k*x1)/m
    return [dx1dt, dx2dt]
x_initial=[x0, Dx0]
t=np.linspace(0, 20, 5000)
sol=odeint(system, x_initial, t)
plt.plot(t, sol[:, 0], label="scipy's odeint method")
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
t2_o=time.time()
print(f'Run time of "odeint method" is {t2_o - t1_o} s')

# Using iteration method by matrices to solve the equation iteratavely
# Solving the homogenous equation
t1_it=time.time()
N=5000
dt=20/(N+1)
# constructing D2=d^2/dt^2 operator 
d1=np.ones(N+2)
d2=np.ones(N+1)
d3=np.ones(N)
D2=(np.diag(d1) + np.diag(-2*d2, -1) + np.diag(d3, -2))/(dt**2)
D2[(0, 1, 1), (0, 0, 1)]=[1., -1/dt, 1/dt]
Id=np.eye(N+2, N+2, k=-1)
Id[1, 0]=0.
L = D2 + (k/m)*Id
g=np.zeros(N+2)
g[:2]=x_initial
x=(np.linalg.inv(L)).dot(g)
t_new=np.linspace(0, 20, N+2)
plt.plot(t_new, x, '--', label='zeroth iteration')

# Using iteration to solve the problem iteratively
n=4 # number or iterations

def modify_g(G, X):
    '''
    A function to modify the inhomogenious part in each iteration.
    '''

    l=(a/(6*m))*(X[1:-1]**3)
    G[2:]=l

for i in range(n):
    modify_g(g, x)
    x=(np.linalg.inv(L)).dot(g)
    plt.plot(t_new, x, ':', label=f'{i+1} th iteration')

t2_it=time.time()
print(f'Run time of "iteration method (by matrices)" is {t2_it - t1_it} s')

# Using Runge-Kutta algorithm to obtain high precision answer
t1_RK=time.time()
t0=0.0
x1_0=x0
x2_0=Dx0
n=5000
h=20/n
def f(t, x):
    x1, x2 = x
    return np.array([x2, (a*x1**3)/(6*m) - (k*x1)/m])
t, x = t0, np.array([x1_0, x2_0])
answer_x1={t0:x1_0}
answer_x2={t0:x2_0}
for i in range(n):
    k1=f(t, x)
    k2=f(t+0.5*h, x + (0.5*h*k1))
    k3=f(t+0.5*h, x + (0.5*h*k2))
    k4=f(t+h, x + (h*k3))
    x = x + ((h/6) * (k1 + 2*k2 + 2*k3 + k4))
    t += h
    answer_x1[t]=x[0]
    answer_x2[t]=x[1]
plt.plot(answer_x1.keys(), answer_x1.values(), '-.', markersize=2, label='Runge-Kutta method')
plt.legend()
plt.axis([0, 20, -10, 10])

t2_RK=time.time()
print(f'Run time of "fourth order Runge-Kutta method" is {t2_RK - t1_RK} s')

t_final=time.time()
print(f'Run time of the program is {t_final - t_initial}')
    
    
    
    
    
    