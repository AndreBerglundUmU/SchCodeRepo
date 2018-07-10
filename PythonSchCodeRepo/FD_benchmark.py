import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from sympy import *
import time
import schroed_schemes as ss
import schroed_functions as sf
import misc_functions as mf
import plot_functions, query_simulation

start_time = time.time()
# Parameters
batch_size = 3
sigma = 1
L = 10*np.pi
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
#u0 = lambda x: 1.4*np.exp(-3*np.power(x,2))

u0FunVal = u0(x)

# Scheme
scheme = ss.FDCN

# Generating Brownian motion and running the simulation
for i in range(batch_size):
	dW = np.random.randn(2,N)*np.sqrt(h/2)
	#result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDFEul,[])
	#result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDBEul,[])
	result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDCN,[])
	result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDMEul,[])
	result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDMP,[])
	result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDStrangSpl,[])
	result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDLieSpl,[])
	result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDFourSpl,[])
	result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDExplExp,[])
	#result = query_simulation.finite_difference_simulation(N,h,deriv_mat_sq,sigma,u0FunVal,dW,ss.FDSymExp,[])
	print(i)
	
end_time = time.time()
print(end_time-start_time)