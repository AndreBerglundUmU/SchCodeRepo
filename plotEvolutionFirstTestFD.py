import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from sympy import *
import time
import schroed_schemes as ss
import schroed_functions as sf
import misc_functions as mf
import plot_functions, query_simulation

# Parameters
sigma = 4
L = 30
M = 2**7
T = 1
N = 2**7
# N = 2^19 for Debussche (T=1)
# M = 2^29 for Debussche
storedTime = 2**6

# Derived values
XInt = [-L,L]
TInt = [0,T]
h = (TInt[1]-TInt[0])/N
dx = (XInt[1]-XInt[0])/M
deriv_mat = mf.create_FD_weight_mat(M,3,2,False)
deriv_mat_sq = deriv_mat.dot(deriv_mat)
t = np.linspace(TInt[0],TInt[1],N+1)
stored_t = np.linspace(TInt[0],TInt[1],storedTime+1)
x = np.linspace(XInt[0],XInt[1],M,endpoint=False)

# Initial function
# u0 = lambda x: 1.4*np.exp(-3*np.power(x,2))
q = 1
alpha = 1
c = 1
sech_fun = lambda x: 2/(np.exp(x) + np.exp(-x))
u0 = lambda x: np.sqrt(2*alpha/q)*np.multiply(np.exp(0.5*1j*c*x),sech_fun(np.sqrt(alpha)*x))
u0 = lambda x: 1.4*np.exp(-3*np.power(x,2))

u0FunVal = u0(x)

# Scheme
scheme = ss.FDLieSpl

# FDFEul
# FDBEul
# FDCN
# FDMEul
# FDMP
# FDStrangSpl
# FDLieSpl
# FDFourSpl
# FDExplExp
# FDSymExp

# Query construction (query, size of return, number of time steps stored, total number of time steps)
my_query = query_simulation.make_query(lambda x: x, M, storedTime, N)
my_query2 = query_simulation.make_query(lambda x: mf.L2_norm(x,dx), 1, storedTime, N)
norm_string = 'L2 norm'

# Generating Brownian motion and running the simulation
dW = np.random.randn(2,N)*np.sqrt(h/2)
result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,scheme,[my_query,my_query2])

# Different plot tests
l2_axis_args = [stored_t[0], stored_t[-1], 0, 1.1*np.max(result[1][:,0])]
plot_functions.plot_waterfall(np.abs(result[0]),x,stored_t,'waterfall_FD.pdf')
time.sleep(3)
plot_functions.plot_norm_evolution(result[1][:,0],stored_t,l2_axis_args,norm_string,'L2_evol_FD.pdf')
time.sleep(1)
#plot_functions.plot_physical_evolution(result[0],x,t,dW,storedTime,N,L)