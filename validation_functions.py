class NotAVectorError(Exception): pass
class NotAMatrixError(Exception): pass
class NotAPositiveIntegerError(Exception): pass
class NotAnOddIntegerError(Exception): pass
class TooLargeValueError(Exception): pass
class XTooLargeError(Exception): pass

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
	except NotAMatrixError as e:
		print(e)
		
def is_boolean(x):
	try:
		is_bool = isinstance(x,bool)
		if not is_bool:
			raise NotABooleanError("Error: Not a boolean")
	except NotABooleanError as e:
		print(e)
		
def is_positive_integer(x):
	try:
		val = int(x)
		if val <= 0:
			raise NotAPositiveIntegerError("Error: Not a positive integer")
	except NotAPositiveIntegerError as e:
		print(e)
		
def is_odd_integer(x):
	try:
		val = int(x)
		if not np.mod(x,2)==1:
			raise NotAnOddIntegerError("Error: Not an odd integer")
	except NotAnOddIntegerError as e:
		print(e)
		
def is_at_most_3(x):
	try:
		val = int(x)
		if val > 3:
			raise TooLargeValueError("Error: Too large value")
	except TooLargeValueError as e:
		print(e)
		
def is_less_than(x,y):
	try:
		if x >= y:
			raise XTooLargeError("Error: X need to be smaller than Y")
	except XTooLargeError as e:
		print(e)