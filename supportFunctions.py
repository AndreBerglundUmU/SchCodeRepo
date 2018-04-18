#! /usr/bin/env python
from scipy.fftpack import fft, ifft
import numpy as np
#	IMPLICITSOLVING
#def CrankNic(currU,dW,k,h):
#def MidPoint(currU,dW,k,h):
#def MidEul(currU,dW,k,h):
#def SymExp(currU,dW,k,h):

def createKVec(leftPoint,rightPoint,numModes):
	halfNumber = numModes
	posModes = np.array(range(halfNumber))
	negModes = np.flip(-np.array(halfNumber))
	tempVec = np.append(posModes,no.arange(1))
	tempVec = np.append(tempVec,negModes)
	k = 2*pi/(rightPoint-leftPoint)*tempVec
	#k = 2*pi/(rightPoint-leftPoint)*[0:numModes/2-1, 0, -numModes/2+1:-1];

def cubicU(currU,sigma):
	np.dot(np.power(np.absolute(currU),2*sigma),currU)

def FEul(currU,dW,k,h,sigma):
	a = (1 - 1j*np.sum(dW)*np.power(k,2));
	b = 1j*h;
	nextU = np.dot(a,currU) + b*fft(cubicU(ifft(currU),sigma))
	return nextU

#def BEul(currU,dW,k,h):
#def StrangSpl(currU,dW,k,h):
#def LieSpl(currU,dW,k,h):
#def FourSpl(currU,dW,k,h):
#def ExplExp(currU,dW,k,h):
