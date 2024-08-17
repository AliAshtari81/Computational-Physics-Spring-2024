#%matplotlib qt
#The above line is to show the plots better (in Spyder), and may not work in some computers. So it can be a comment.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint #for checking the answers of second odrer equations
# x=[0-30] with step=0.05
dx=0.05
x=np.arange(0, 30+dx, dx)
N=len(x)-2
# constructing D=d/dx (I make it just for first order equations, to adapt with one initial condition.)
# Note that for second order equations I will modify it a little bit to adapt with two initial conditions.
d1=np.ones(N+2)
d2=np.ones(N+1)
D=np.diag(d1/dx) + np.diag(-d2/dx, -1)
D[0, 0]=1.
# constructing D1=d/dx (This one is also first derivative, but for second order equations)
D1=np.diag(d1/dx) + np.diag(-d2/dx, -1)
D1[(0, 1, 1), (0, 0, 1)]=0.
# Note that I set the first two rows of D1 to zero, because I construct this operator specifiecly for second order equations, and in this equations I prefer to apply all of initial conditions of D2.
# constructing D2=d^2/dx^2 
d3=np.ones(N)
D2=np.diag(d1/dx**2) + np.diag(-2*d2/dx**2, -1) + np.diag(d3/dx**2, -2)
D2[0, 0]=1.
D2[1, 0]=-1/dx
D2[1, 1]=1/dx
# Now using these operators for given examples.

# Example 1 : df/dx=-2x with inintial condition: f(0)=0
# constructing g_tilda=[f0, g(x)=-2x]
f1_0=np.array([0.]) #initial value
g1=np.concatenate((f1_0, np.vectorize(lambda x: -2*x)(x[:-1])))
# solving the equation
f1=np.matmul(np.linalg.inv(D), g1)
plt.figure()
plt.plot(x, f1, label='numerical solution for f(x)')
plt.plot(x, -x**2, '-.', label="analytical solution for f(x)")
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r'Example 1 _ Solution of $\frac{df}{dx} = -2x$ with initial value: f(0)=0')
plt.grid()
plt.show()

# Example 2 : df/dx=f(x)/4 with initial condition: f(0)=1
# rearanging the equation: df/dx - f(x)/4= 1* delta(x)
f2_0=1. #initial value
L2=D - 0.25*np.diag(d2, -1) # total operator acting on f(x)
delta2=np.zeros(N+2)
delta2[0]=f2_0
# solving the equation
f2=np.matmul(np.linalg.inv(L2), delta2)
plt.figure()
plt.plot(x, f2, label='numerical solution for f(x)')
plt.plot(x, np.exp(x/4), '-.', label='analytical solution for f(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r'Example 2 _ Solution of $\frac{df}{dx} = \frac{1}{4} f(x)$ with initial value: f(0)=1')
plt.grid()
plt.show()

# Example 3 : d^2f/dx^2 + f(x) = 0 with initial conditions: f(0)=1 & f'(0)=0
# rewriting the equation as d^2f/dx^2 + f(x) = delta(x)
f3_0=[1., 0.]
I=np.diag(d2, -1)
I[1, 0]=0.
L3= D2 + I # total operator acting on f(x)
delta3=np.zeros(N+2)
delta3[0:2]=f3_0
# solving the equation
f3=np.matmul(np.linalg.inv(L3), delta3)
plt.figure()
plt.plot(x, f3, label='numerical solution for f(x)')
plt.plot(x, np.cos(x), '-.', label='analytical solution for f(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r"Example 3 _ Solution of $\frac{d^2 f}{dx^2} + f(x) = 0$ with initial value: f(0)=1 & f'(0)=0")
plt.grid()
plt.show()

# Example 4 : d^2f/dx^2 + gamma * df/dx + f(x) = 0 with initial conditons: f(0)=1 & f'(0)=0
# rewriting the equation as d^2f/dx^2 + gamma * df/dx + f(x) = delta(x)
f4_0=[1., 0.]
gamma=0.4
L4=D2 + gamma*D1 + I #I is defined in line 64 & 65
delta4=np.zeros(N+2)
delta4[0:2]=f4_0
# solving the equation
f4=np.matmul(np.linalg.inv(L4), delta4)
# using odeint for sanity check
def system(y, x):
    y1, y2 = y
    dy1dx = y2
    dy2dx = -gamma*y2 - y1
    return [dy1dx, dy2dx]
y_initial=[1., 0.]
sol4=odeint(system, y_initial, x)
#ploting
plt.figure()
plt.plot(x, f4, label='numerical solution for f(x)')
plt.plot(x, sol4[:, 0], '-.', label="numerical solution using scipy's odeint")
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r"Example 4 _ Solution of $\frac{d^2 f}{dx^2} + 0.4 \frac{df}{dx} +  f(x) = 0$ with initial value: f(0)=1 & f'(0)=0")
plt.grid()
plt.show()

# Example 5 : d^2f/dx^2 + gamma * df/dx + f(x) = g(x) with g(x)=sin(x) with initial conditons: f(0)=1 & f'(0)=0
f5_0=[1., 0.]
gamma=0.5
L5=D2 + gamma*D1 + I #I is defined in line 64 & 65
g_tilda=np.concatenate((np.array(f5_0), np.vectorize(lambda x: np.sin(x))(x[1:-1])))
# solving the equation
f5=np.matmul(np.linalg.inv(L5), g_tilda)
# using odeint for sanity check
def system(y, x):
    y1, y2 = y
    dy1dx = y2
    dy2dx = -gamma*y2 - y1 + np.sin(x)
    return [dy1dx, dy2dx]
y_initial=[1., 0.]
sol5=odeint(system, y_initial, x)
#ploting
plt.figure()
plt.plot(x, f5, label='numerical solution for f(x)')
plt.plot(x, sol5[:, 0], '-.', label="numerical solution using scipy's odeint")
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r"Example 5 _ Solution of $\frac{d^2 f}{dx^2} + 0.5 \frac{df}{dx} +  f(x) = sin(x)$ with initial value: f(0)=1 & f'(0)=0")
plt.grid()
plt.show()