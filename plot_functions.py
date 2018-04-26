#! /usr/bin/env python
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import validation_functions as vf

# import randomstate as rnd
# Might be necessary for parallel rng generation
# https://pypi.org/project/randomstate/1.10.1/

# example
# workers.simulation(L,M,T,N,sigma,u0Fun,schemes.PSLieSpl,[queries]) if queries is one query

# import importlib
# importlib.reload(...)

def plot_physical_evolution(function_evolution,x,t,dW,storedTime,N,L):
	vf.is_vector(x)
	# vf.is_matrix(function_evolution) # function_evolution is not a numpy matrix
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plotVal = ifft(function_evolution[0][0,:])
	axis_ceil = 1.1*np.max(np.abs(plotVal))
	line = ax.plot(x, np.real(plotVal),'b', x, np.imag(plotVal),'r', x, np.abs(plotVal), 'y')

	W = np.cumsum(sum(dW))
	W = np.insert(W,0,0)
	for i in range(storedTime):
		index = int((i+1)*N/storedTime)
		t_val = t[index]
		W_val = W[index]
		
		plotVal = ifft(function_evolution[0][i+1,:])
		line[0].set_ydata(np.real(plotVal))
		line[1].set_ydata(np.imag(plotVal))
		line[2].set_ydata(np.abs(plotVal))
		#plt.plot(x, np.real(plotVal),'b', x, np.imag(plotVal),'r', x, np.abs(plotVal), 'y')
		plt.axis([-L, L, -axis_ceil, axis_ceil])
		plt.suptitle('t = {:1.4f}, W_t = {:4.4f}'.format(t_val,W_val))#,fontsize=20)
		#plt.draw()
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(np.spacing(1))

	plt.show()