#! /usr/bin/env python
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import axes3d
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

def plot_waterfall(function_evolution,x,stored_t):
# https://stackoverflow.com/questions/31189665/plotting-using-polycollection-in-matplotlib
	# fun_val = np.abs(function_evolution)
	fig = plt.figure(figsize=(15,15))
	ax = fig.add_subplot(111, projection='3d')
	verts = []
	for i in range(len(stored_t)):
		xs = np.concatenate([[x[0]],x, [x[-1]]])
		ys = np.concatenate([[0],function_evolution[i,:],[0]])
		verts.append(list(zip(xs, ys)))
	poly = PolyCollection(verts, facecolors=(1,1,1,1), edgecolors=(0,0,1,1))

	# The zdir keyword makes it plot the "z" vertex dimension (radius)
	# along the y axis. The zs keyword sets each polygon at the
	# correct radius value.
	ax.add_collection3d(poly, zs=stored_t, zdir='y')

	ax.set_xlim3d(x.min(), x.max())
	ax.set_xlabel('x')
	ax.set_ylim3d(stored_t.min(), stored_t.max())
	ax.set_ylabel('t')
	ax.set_zlim3d(function_evolution.min(), function_evolution.max())
	ax.set_zlabel('|u|')
	#plt.show()
	fig.canvas.draw()
	plt.savefig('waterfall2.pdf')
	return fig
	
def plot_norm_evolution(norm_evolution,stored_t,axis_args,norm_string):
	vf.is_vector(norm_evolution)
	# vf.is_matrix(function_evolution) # function_evolution is not a numpy matrix
	fig = plt.figure(figsize=(15,15))
	plt.plot(stored_t,np.real(norm_evolution))
	plt.axis(axis_args)
	plt.xlabel('t')
	plt.ylabel(norm_string)
	#plt.show()
	plt.savefig('plot2.pdf')
	return fig
	
def plot_physical_evolution(function_evolution,x,t,dW,storedTime,N,L):
	vf.is_vector(x)
	# vf.is_matrix(function_evolution) # function_evolution is not a numpy matrix
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plotVal = function_evolution[0,:]
	axis_ceil = 1.1*np.max(np.abs(plotVal))
	line = ax.plot(x, np.real(plotVal),'b', x, np.imag(plotVal),'r', x, np.abs(plotVal), 'y')

	W = np.cumsum(sum(dW))
	W = np.insert(W,0,0)
	for i in range(storedTime):
		index = int((i+1)*N/storedTime)
		t_val = t[index]
		W_val = W[index]
		
		plotVal = function_evolution[i+1,:]
		line[0].set_ydata(np.real(plotVal))
		line[1].set_ydata(np.imag(plotVal))
		line[2].set_ydata(np.abs(plotVal))
		plt.legend(line, ['real(u)','imag(u)','abs(u)'])
		#plt.plot(x, np.real(plotVal),'b', x, np.imag(plotVal),'r', x, np.abs(plotVal), 'y')
		plt.axis([-L, L, -axis_ceil, axis_ceil])
		plt.suptitle('t = {:1.4f}, W_t = {:4.4f}'.format(t_val,W_val))#,fontsize=20)
		#plt.draw()
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(np.spacing(1))

	plt.show()