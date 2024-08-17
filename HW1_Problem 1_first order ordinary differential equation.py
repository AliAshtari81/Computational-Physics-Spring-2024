#%matplotlib qt
#The above line is to show the plots better (in Spyder), and may not work in some computers. So it can be a comment.
import numpy as np
import matplotlib.pyplot as plt
# solving df/dt=cos(t) with initial condition f(0)=1
# t=[0-15]s
dt=0.1
t=np.arange(0, 15+dt, dt)
N=len(t)-2 # N points in between, two points at the endpoints
# constructing the operator D=d/dt
d1=np.ones(N+2)
d2=np.ones(N+1)
D=np.diag(d1/dt) + np.diag(-d2/dt, -1)
D[0, 0]=1.
# constructing g_tilda=[f0, g(t)=cos(t)]
f0=np.array([1.])
g_tilda=np.concatenate((f0, np.vectorize(lambda t: np.cos(t))(t[:-1])))
#solving the equation
y=np.matmul(np.linalg.inv(D), g_tilda)
plt.plot(t, y, label='y(t)')
plt.plot(t, 1+np.sin(t), '-.', label='analytical answer for y(t)') #It is obvious that the analytical answer is y(t)=1+sin(t), by running this line my numerical answer can be checked with analytical answer and it is obvious that they are equal.
#plotting g(t)=cos(t)
plt.plot(t, np.cos(t), label='g(t)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r'Solution of $\frac{df}{dx} = cos(x)$ with initial value: f(0)=1')
plt.grid()
plt.legend()
plt.show()