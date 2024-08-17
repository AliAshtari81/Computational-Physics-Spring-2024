#%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt

n_sampling=int(1e8)
n_warmup=int(n_sampling/5)
dx=0.1
#%%
#
def p(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)

x_p=np.zeros(n_sampling)
x_p[0]=np.random.randn()
for i in range(1, n_sampling): # generating x with normal distribution p(x)
    x0=x_p[i-1]
    delta_x=dx*np.random.randn()
    x1 = x0 + delta_x
    r10 = p(x1)/p(x0)
    if r10> np.random.randn():
        x_p[i] = x1
    else:
        x_p[i] = x0
x_p = x_p[n_warmup:]
        
# a)

w = lambda x: np.exp(-abs(x))
x_w=np.zeros(n_sampling)
x_w[0]=np.random.randn()
for i in range(1, n_sampling): # generating x with w(x) distribution
    x0=x_w[i-1]
    delta_x=dx*np.random.randn()
    x1 = x0 + delta_x
    r10 = w(x1)/w(x0)
    if r10> np.random.randn():
        x_w[i] = x1
    else:
        x_w[i] = x0
x_w = x_w[n_warmup:]
# Normalizing w(x)
w_tilda = lambda x: w(x)/p(x)
W_tilda_mean=np.mean(w_tilda(x_p))

# Calculating the mean with Monte-Carlo
f1_avg_MC=(np.mean(x_w))/W_tilda_mean
print(f'Monte-Carlo : <f1(x)> = {f1_avg_MC}')
# Calculating the mean with Riemann sum
X=np.arange(-5, 5+dx, dx)
f1_avg_RS=(np.sum(X * w(X)))/(np.sum(w(X)))
print(f'Riemann Sum : <f1(x)> = {f1_avg_RS}')

# b)
#%%
w_b = lambda x: 1 / (1 + x**2)
x_w=np.zeros(n_sampling)
flag=True
while flag==True: # e mechanism to make sure initial point is inside the interval
    x_w[0]=np.random.randn()
    if -10*np.pi<=x_w[0]<=15*np.pi:
        flag=False
    else:
        flag=True
for i in range(1, n_sampling): # generating x with w(x) distribution
    x0=x_w[i-1]
    delta_x=dx*np.random.randn()
    x1 = x0 + delta_x
    r10 = w_b(x1)/w_b(x0)
    if r10> np.random.randn():
        x_w[i] = x1
    else:
        x_w[i] = x0
x_w = x_w[n_warmup:]
# Interval of integration is -10 to 10, because integral of f(x)w(x) is divergent from -infinity to infinity
x_w = x_w[abs(x_w)<=20]
# Normalizing w(x)
w_tilda_b = lambda x: w_b(x)/p(x)
W_tilda_mean=np.mean(w_tilda_b(x_p))
x=np.arange(-20, 20+dx, dx)
W=np.sum(w_b(x))*dx

# Calculating the mean with Monte-Carlo
f2_avg_MC=(np.mean(x_w**2))/W
print(f'Monte-Carlo : <f2(x)> = {f2_avg_MC}')
# Calculating the mean with Riemann sum
X=np.arange(-10, 10+dx, dx)
f2_avg_RS=(np.sum(X**2 * w_b(X)))/(np.sum(w_b(X)))
print(f'Riemann Sum : <f2(x)> = {f2_avg_RS}')

#%%
#c)
w_c = lambda x: abs(np.cos(x))
x_w=np.zeros(n_sampling)
flag=True
while flag==True: # e mechanism to make sure initial point is inside the interval
    x_w[0]=np.random.randn()
    if -10*np.pi<=x_w[0]<=15*np.pi:
        flag=False
    else:
        flag=True
for i in range(1, n_sampling): # generating x with w(x) distribution
    x0=x_w[i-1]
    delta_x=dx*np.random.randn()
    x1 = x0 + delta_x
    r10 = w_c(x1)/w_c(x0)
    if r10> np.random.randn():
        x_w[i] = x1
    else:
        x_w[i] = x0
x_w = x_w[n_warmup:]
# Interval of integration is -10pi to 15pi, because integral of f(x)w(x) is divergent from -infinity to infinity
mask = (x_w >= -10*np.pi) & (x_w <= 15*np.pi)
x_w = x_w[mask]

numerator = np.mean(x_w**2 * np.sign(np.cos(x_w)))
denominator = np.mean(np.sign(np.cos(x_w)))

# Calculating the mean with Monte-Carlo
f3_avg_MC=numerator / denominator
print(f'Monte-Carlo : <f3(x)> = {f3_avg_MC}')
# Calculating the mean with Riemann sum
X=np.arange(-10*np.pi, 15*np.pi+dx, dx)
f3_avg_RS=(np.sum(X**2 * np.cos(X)))/(np.sum(np.cos(X)))
print(f'Riemann Sum : <f3(x)> = {f3_avg_RS}')

#%%
#d)
w_d=lambda x: np.cos(x) * np.exp(-x**2 /2)
p_d = lambda x: abs(np.cos(x)) * np.exp(-x**2 /2)
x_w=np.zeros(n_sampling)
x_w[0]=np.random.randn()
for i in range(1, n_sampling): # generating x with w(x) distribution
    x0=x_w[i-1]
    delta_x=dx*np.random.randn()
    x1 = x0 + delta_x
    r10 = p_d(x1)/p_d(x0)
    if r10> np.random.randn():
        x_w[i] = x1
    else:
        x_w[i] = x0
x_w = x_w[n_warmup:]
numerator = np.mean(x_w**2 * np.sign(w_d(x_w)))
denominator = np.mean(np.sign(w_d(x_w)))

# Calculating the mean with Monte-Carlo
f4_avg_MC=numerator / denominator
print(f'Monte-Carlo : <f4(x)> = {f4_avg_MC}')
# Calculating the mean with Riemann sum
X=np.arange(-10*np.pi, 15*np.pi+dx, dx)
f4_avg_RS=(np.sum(X**2 * w_d(X)))/(np.sum(w_d(X)))
print(f'Riemann Sum : <f4(x)> = {f4_avg_RS}')

#%%
#e)
w_e=lambda x: np.exp(-x**2 /2)
x_w=np.zeros(n_sampling)
x_w[0]=np.random.randn()
for i in range(1, n_sampling): # generating x with w(x) distribution
    x0=x_w[i-1]
    delta_x=dx*np.random.randn()
    x1 = x0 + delta_x
    r10 = w_e(x1)/w_e(x0)
    if r10> np.random.randn():
        x_w[i] = x1
    else:
        x_w[i] = x0
x_w = x_w[n_warmup:]

# Calculating the mean with Monte-Carlo
f5_avg_MC= (np.mean(np.cos(10*x_w)*w_e(x_w)))/(np.sqrt(2*np.pi))
print(f'Monte-Carlo : <f5(x)> = {f5_avg_MC}')
# Calculating the mean with Riemann sum
X=np.arange(-10, 10+dx, dx)
f5_avg_RS=(np.sum(np.cos(10*X) * w_e(X)))/(np.sum(w_e(X)))
print(f'Riemann Sum : <f5(x)> = {f5_avg_RS}')
