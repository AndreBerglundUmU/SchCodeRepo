#! /usr/bin/env python
from scipy.fftpack import fft, ifft
import numpy as np
import supportFunctions as sf
import schemes

# import randomstate as rnd
# Might be necessary for parallel rng generation
# https://pypi.org/project/randomstate/1.10.1/

# example
# workers.simulation(L,M,T,N,sigma,u0Fun,schemes.PSLieSpl,[queries]) if queries is one query

# import importlib
# importlib.reload(...)

class Query(object):
	# This class will assume that N is a power of 2 and that the storage size is as well
	
	def __init__(self, function, spaceStorageSize, desiredTimeStorageSize,N):
		self.function = function
		self.spaceStorageSize = spaceStorageSize
		self.N = N
		if desiredTimeStorageSize >= N:
			self.periodicity = 1
			self.timeStorageSize = N + 1
		else:
			self.periodicity = int(N / desiredTimeStorageSize)
			self.timeStorageSize = desiredTimeStorageSize + 1
	
	def preallocateQueryResult(self):
		return np.zeros(shape=(self.timeStorageSize,self.spaceStorageSize),dtype=np.complex_)
		

def make_Query(function, spaceStorageSize, desiredTimeStorageSize,N):
    query = Query(function, spaceStorageSize, desiredTimeStorageSize,N)
    return query


def simulation(L,M,T,N,sigma,u0,scheme,queries):
	XInt = [-L,L]
	TInt = [0,T]
	h = (TInt[1]-TInt[0])/N
	dx = (XInt[1]-XInt[0])/M
	k = sf.createKVec(XInt[1],XInt[0],M)
	kSq = np.power(k,2)
	t = np.linspace(TInt[0],TInt[1],N+1)
	x = np.linspace(XInt[0],XInt[1],M,endpoint=False)
	
	# Need to preallocate memory for query results
	currU = u0(x)
	queryStorage = []
	queryIndex = []
	for q in range(len(queries)):
		queryStorage.append(queries[q].preallocateQueryResult())
		queryStorage[q][0,:] = queries[q].function(currU)
		queryIndex.append(1)
	
	for i in range(N):
		dW = np.random.randn(2,1)*(h/2)
		currU = scheme(currU,dW,k,kSq,h,sigma)
		for q in range(len(queries)):
			if (i % queries[q].periodicity) == (queries[q].periodicity - 1):
				queryStorage[q][queryIndex[q],:] = queries[q].function(currU)
				queryIndex[q] += 1
			
	return queryStorage