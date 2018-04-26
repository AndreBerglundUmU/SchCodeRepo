class NotAVectorError(Exception): pass
class NotAMatrixError(Exception): pass

import numpy as np

def is_vector(x):
	num_dim = len(x.shape)
	try:
		if not num_dim==1:
			raise NotAVectorError("Error: Not a vector")
	except NotAVectorError as e:
		print(e)
		
def is_matrix(x):
	num_dim = len(x.shape)
	try:
		if not num_dim==2:
			raise NotAMatrixError("Error: Not a matrix")
	except NotAVectorError as e:
		print(e)