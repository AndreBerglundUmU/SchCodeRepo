#! /usr/bin/env python
import numpy as np
import validation_functions as vf

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

def make_query(function,spaceStorageSize,desiredTimeStorageSize,N):
    query = Query(function, spaceStorageSize, desiredTimeStorageSize,N)
    return query

def pseudospectral_simulation(N,h,kSq,sigma,u0,W,scheme,queries):
	vf.is_vector(u0)
	vf.is_vector(kSq)
	# Need to preallocate memory for query results
	currU = u0
	queryStorage = []
	queryIndex = []
	for q in range(len(queries)):
		queryStorage.append(queries[q].preallocateQueryResult())
		queryStorage[q][0,:] = queries[q].function(currU)
		queryIndex.append(1)
	
	for i in range(N):
		dW = W[:,i]
		# dW = np.random.randn(2,1)*(h/2)
		currU = scheme(currU,dW,kSq,h,sigma)
		for q in range(len(queries)):
			if (i % queries[q].periodicity) == (queries[q].periodicity - 1):
				queryStorage[q][queryIndex[q],:] = queries[q].function(currU)
				queryIndex[q] += 1
			
	return queryStorage
	
def finite_difference_simulation(N,h,FDMatSq,sigma,u0,W,scheme,queries):
	vf.is_vector(u0)
	# Need to preallocate memory for query results
	currU = u0
	queryStorage = []
	queryIndex = []
	for q in range(len(queries)):
		queryStorage.append(queries[q].preallocateQueryResult())
		queryStorage[q][0,:] = queries[q].function(currU)
		queryIndex.append(1)
	
	for i in range(N):
		dW = W[:,i]
		# dW = np.random.randn(2,1)*(h/2)
		currU = scheme(currU,dW,FDMatSq,h,sigma)
		for q in range(len(queries)):
			if (i % queries[q].periodicity) == (queries[q].periodicity - 1):
				print(currU)
				print(type(currU))
				queryStorage[q][queryIndex[q],:] = queries[q].function(currU)
				queryIndex[q] += 1
			
	return queryStorage