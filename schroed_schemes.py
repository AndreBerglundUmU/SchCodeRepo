#! /usr/bin/env python
from scipy.fftpack import fft, ifft
import numpy as np
import schroed_functions as sf

def PSFEul(currU,dW,k,kSq,h,sigma):
	a = (1 - 1j*np.sum(dW)*kSq);
	b = 1j*h;
	nextU = np.multiply(a,currU) + b*fft(sf.cubicU(ifft(currU),sigma))
	return nextU

def PSBEul(currU,dW,k,kSq,h,sigma):
	a = 1/(1 + 1j*np.sum(dW)*kSq);
	b = 1j*h/(1 + 1j*np.sum(dW)*kSq);
	nextU = np.multiply(a,currU) + np.multiply(b,fft(sf.cubicU(ifft(currU),sigma)))
	return nextU

def PSCN(currU,dW,k,kSq,h,sigma):
	nextU = sf.PSEulTypeSolver(currU,sum(dW),k,kSq,h,sigma,sf.CNNonLin)
	return nextU
	
def PSMEul(currU,dW,k,kSq,h,sigma):
	nextU = sf.PSEulTypeSolver(currU,sum(dW),k,kSq,h,sigma,sf.MEulNonLin)
	return nextU
	
def PSMP(currU,dW,k,kSq,h,sigma):
	nextU = sf.PSEulTypeSolver(currU,sum(dW),k,kSq,h,sigma,sf.MPNonLin)
	return nextU
	
def PSStrangSpl(currU,dW,k,kSq,h,sigma):
	# Half derivative step
	tempPhysCurrU = ifft(np.multiply(np.exp(-dW[0]*1j*kSq),currU))
	# Full nonlinear step
	tempU = fft(np.multiply(np.exp(h*1j*np.power(np.abs(tempPhysCurrU),2*sigma)),tempPhysCurrU))
	# Half derivative step
	nextU = np.multiply(np.exp(-dW[1]*1j*kSq),tempU)
	return nextU
	
def PSLieSpl(currU,dW,k,kSq,h,sigma):
	# Full derivative step
	tempPhysCurrU = ifft(np.multiply(np.exp(-sum(dW)*1j*kSq),currU))
	# Full nonlinear step
	nextU = fft(np.multiply(np.exp(h*1j*np.power(np.abs(tempPhysCurrU),2*sigma)),tempPhysCurrU))
	return nextU
	
def PSFourSpl(currU,dW,k,kSq,h,sigma):
	# Full nonlinear step
	tempPhysCurrU = ifft(currU)
	tempU = fft(np.multiply(np.exp(h*1j*np.power(np.abs(tempPhysCurrU),2*sigma)),tempPhysCurrU))
	# Full derivative step
	nextU = np.multiply(np.exp(-sum(dW)*1j*kSq),tempU)
	return nextU
	
def PSExplExp(currU,dW,k,kSq,h,sigma):
	nextU = np.multiply(np.exp(-sum(dW)*1j*kSq),currU) + 1j*h*np.multiply(np.exp(-sum(dW)*1j*kSq),fft(sf.cubicU(ifft(currU),sigma)))
	return nextU
	
def PSSymExp(currU,dW,k,kSq,h,sigma):
	kSquared = kSq;
	NStar = sf.PSNStarSolver(currU,dW[0],k,kSq,h,sigma)
	nextU = np.multiply(np.exp(-sum(dW)*1j*kSq),currU) + np.multiply(h*np.exp(-dW[1]*1j*kSq),NStar)
	return nextU