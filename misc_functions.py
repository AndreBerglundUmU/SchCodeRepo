import numpy as np
import validation_functions as vf


def createKVec(leftPoint,rightPoint,numModes):
	halfNumber = int(numModes/2)
	posModes = np.array(range(halfNumber))
	negModes = np.flip(-np.array(range(1,halfNumber)),0)
	tempVec = np.append(posModes,np.arange(1))
	tempVec = np.append(tempVec,negModes)
	return 2*np.pi/(rightPoint-leftPoint)*tempVec

def trapezoidalIntegral(u,dx):
	vf.is_vector(u)
	return dx/2*np.sum(u[:-1] + u[1:])
	
def L2Norm(u,dx):
	vf.is_vector(u)
	int_val = np.power(np.abs(u),2)
	return trapezoidalIntegral(int_val,dx)

def H1Norm(u,dx):
	vf.is_vector(u)
	deriv_val = (u[1:]-u[:-1])/dx	
	return L2Norm(u,dx) + L2Norm(np.append(deriv_val,deriv_val[-1]),dx)
