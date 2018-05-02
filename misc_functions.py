import numpy as np
import validation_functions as vf
from scipy import sparse, linalg


def create_k_vec(leftPoint,rightPoint,numModes):
	halfNumber = int(numModes/2)
	posModes = np.array(range(halfNumber))
	negModes = np.flip(-np.array(range(1,halfNumber)),0)
	tempVec = np.append(posModes,np.arange(1))
	tempVec = np.append(tempVec,negModes)
	return 2*np.pi/(rightPoint-leftPoint)*tempVec

def create_FD_weight_vector(num_deriv_points,deriv_num):
	vf.is_positive_integer(deriv_num)
	vf.is_positive_integer(num_deriv_points)
	vf.is_odd_integer(num_deriv_points)
	
	bound = (num_deriv_points-1)/2
	row = np.arange(-bound,bound+1)

	mat = np.zeros((num_deriv_points,num_deriv_points))

	for i in range(num_deriv_points):
		mat[i,:] = np.power(row,i)

	vec = np.zeros((num_deriv_points,1))
	vec[deriv_num] = np.math.factorial(deriv_num)

	weight = linalg.solve(mat,vec)
	return weight
	
def create_FD_weight_mat(M,num_deriv_points,deriv_num,periodic):
	vf.is_positive_integer(M)
	vf.is_positive_integer(num_deriv_points)
	vf.is_odd_integer(num_deriv_points)
	vf.is_less_than(num_deriv_points,M)
	vf.is_boolean(periodic)
	
	if not periodic:
		# Can only accept 3 points at most
		vf.is_at_most_3(num_deriv_points)
	
	# FD weight vector
	weight_vector = create_FD_weight_vector(num_deriv_points,deriv_num)
	# Calculations will assume periodic, until the end. This will wrap some elements
	no_wrap = int((num_deriv_points-1)/2)
	
	# Main diagonal which have zero elements as first an last elements if non-periodic
	if periodic:
		insert_vector_1 = weight_vector[no_wrap]*np.ones(M)
	else:
		insert_vector_1 = np.concatenate((np.array([0]),weight_vector[no_wrap]*np.ones(M-2),np.array([0])))
	weight_mat = sparse.diags(insert_vector_1, 0)
	
	for i in range(no_wrap):
		# Add the other large diagonals, which have zero elements as first and last elements if non-periodic
		if periodic:
			insert_vector_1 = weight_vector[no_wrap+(i+1)]*np.ones(M-(i+1))
			insert_vector_2 = weight_vector[no_wrap-(i+1)]*np.ones(M-(i+1))
		else:
			insert_vector_1 = np.concatenate((np.array([0]),weight_vector[no_wrap+(i+1)]*np.ones(M-(i+3)),np.array([0])))
			insert_vector_2 = np.concatenate((np.array([0]),weight_vector[no_wrap-(i+1)]*np.ones(M-(i+3)),np.array([0])))
		weight_mat += sparse.diags(insert_vector_1, i+1)
		weight_mat += sparse.diags(insert_vector_2, -(i+1))
		# Add values which wrap around if periodic
		if periodic:
			weight_mat += sparse.diags(weight_vector[no_wrap+(i+1)]*np.ones(i+1), -(M-(i+1)))
			weight_mat += sparse.diags(weight_vector[no_wrap-(i+1)]*np.ones(i+1), M-(i+1))
	
	return sparse.csc_matrix(weight_mat)
	
def trapezoidal_integral(u,dx):
	vf.is_vector(u)
	return dx/2*np.sum(u[:-1] + u[1:])
	
def L2_norm(u,dx):
	vf.is_vector(u)
	int_val = np.power(np.abs(u),2)
	return trapezoidal_integral(int_val,dx)

def H1_norm(u,dx):
	vf.is_vector(u)
	deriv_val = (u[1:]-u[:-1])/dx	
	return L2_norm(u,dx) + L2_norm(np.append(deriv_val,deriv_val[-1]),dx)
