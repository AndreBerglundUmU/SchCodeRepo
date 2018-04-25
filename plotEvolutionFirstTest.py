import numpy as np
import schemes, supportFunctions as sf, workers
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from sympy import *
import plotTools

# Parameters
sigma = 1
L = 30
M = 2**10
T = 1
N = 2**10
storedTime = 2**8

# Derived values
XInt = [-L,L]
TInt = [0,T]
h = (TInt[1]-TInt[0])/N
dx = (XInt[1]-XInt[0])/M
k = sf.createKVec(XInt[1],XInt[0],M)
kSq = np.power(k,2)
t = np.linspace(TInt[0],TInt[1],N+1)
x = np.linspace(XInt[0],XInt[1],M,endpoint=False)

# Initial function
# u0 = lambda x: 1.4*np.exp(-3*np.power(x,2))
q = 1
alpha = 1
c = 1
sech_fun = lambda x: 2/(np.exp(x) + np.exp(-x))
u0 = lambda x: np.sqrt(2*alpha/q)*np.multiply(np.exp(0.5*1j*c*x),sech_fun(np.sqrt(alpha)*x))

u0FunVal = fft(u0(x))

# Scheme
scheme = schemes.PSLieSpl

# Query construction (query, size of return, number of time steps stored, total number of time steps)
my_query = workers.make_query(lambda x: x, M, storedTime, N)

# Generating Brownian motion and running the simulation
dW = np.random.randn(2,N)*np.sqrt(h/2)
query_result = workers.simulation(N,h,k,kSq,sigma,u0FunVal,dW,scheme,[my_query])


# Plot over time
plotTools.plotRunning(query_result,x,t,dW,storedTime,N,L)