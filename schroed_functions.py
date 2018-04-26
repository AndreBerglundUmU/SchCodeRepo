#! /usr/bin/env python
from scipy.fftpack import fft, ifft
import numpy as np
import validation_functions as vf
	
def cubicU(currU,sigma):
	vf.is_vector(currU)
	return np.multiply(np.power(np.absolute(currU),2*sigma),currU)
	
def PSEulTypeSolver(currU,dW,k,kSq,h,sigma,G):
	vf.is_vector(currU)
	# Initialize loop variables
	crit = True;
	i = 1;
	nextU = currU;
	realSpaceCurrU = ifft(currU);
	realSpaceNextU = realSpaceCurrU;
	# Calculate some values which don't change
	a = np.multiply(
		np.divide(1 - 1j*dW/2*kSq,
			1 + 1j*dW/2*kSq),
		currU)
	b = 1j*h/(1 + 1j*dW/2*kSq);

	while (crit and i < 120):
		tempU = nextU;
		nextU = a + np.multiply(b,fft(G(realSpaceCurrU,realSpaceNextU,sigma)));
		realSpaceNextU=ifft(nextU);
		crit = np.linalg.norm(tempU-nextU) > np.spacing(1);
		i = i+1;
	return nextU

def PSNStarSolver(currU,firstHalfdW,k,kSq,h,sigma):
	vf.is_vector(currU)
	# Initialize loop variables
	crit = True
	i = 1
	NStar = currU
	# Calculate some values which don't change
	a = np.multiply(
		np.exp(-firstHalfdW*1j*kSq),
		currU)

	while crit and i < 120:
		oldNStar = NStar
		tempNStar = a + h/2*NStar
		NStar = fft(1j*cubicU(ifft(tempNStar),sigma))
		crit = np.linalg.norm(oldNStar-NStar) > np.spacing(1);
		i = i+1;
	return NStar
	
def CNNonLin(u,v,sigma):
	#vf.is_vector(u) #how expensive?
	#vf.is_vector(v)
	if sigma == 1:
		retVal = np.multiply(
			np.power(np.abs(u),2) + 
			np.power(np.abs(v),2),
			u+v)/(2*(sigma+1))
	elif sigma == 2:
		retVal = np.multiply(
			np.power(np.abs(u),4) + 
			np.multiply(np.power(np.abs(u),2),np.power(np.abs(v),2)) + 
			np.power(np.abs(v),4),
			u+v)/(2*(sigma+1))
	elif sigma == 3:
		retVal = np.multiply(
			np.power(np.abs(u),6) + 
			np.multiply(np.power(np.abs(u),4),np.power(np.abs(v),2)) + 
			np.multiply(np.power(np.abs(u),2),np.power(np.abs(v),4)) + 
			np.power(np.abs(v),6),
			u+v)/(2*(sigma+1))
	elif sigma == 4:
		retVal = np.multiply(
			np.power(np.abs(u),8) + 
			np.multiply(np.power(np.abs(u),6),np.power(np.abs(v),2)) + 
			np.multiply(np.power(np.abs(u),4),np.power(np.abs(v),4)) + 
			np.multiply(np.power(np.abs(u),2),np.power(np.abs(v),6)) + 
			np.power(np.abs(v),8),
			u+v)/(2*(sigma+1))
	else:
		retVal = np.multiply(
			np.divide(
				np.power(np.abs(u),2*(sigma+1)) - np.power(np.abs(v),2*(sigma+1)),
				np.power(np.abs(u),2) - np.power(np.abs(v),2) + pow(np.spacing(0),1/2)), # Adding square root of smallest value to interpret as nonzero
			u+v)/(2*(sigma+1))
	return retVal

def MEulNonLin(u,v,sigma):
	#vf.is_vector(u) #how expensive?
	retVal = cubicU(u,sigma)
	return retVal

def MPNonLin(u,v,sigma):
	#vf.is_vector(u) #how expensive?
	#vf.is_vector(v)
	retVal = cubicU((u+v)/2,sigma)
	return retVal
