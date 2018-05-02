#! /usr/bin/env python
from scipy.fftpack import fft, ifft
from scipy import sparse, linalg
import numpy as np
import schroed_functions as sf

def PSFEul(currU,dW,kSq,h,sigma):
	a = (1 - 1j*np.sum(dW)*kSq);
	b = 1j*h;
	nextU = np.multiply(a,currU) + b*fft(sf.cubicU(ifft(currU),sigma))
	return nextU

def PSBEul(currU,dW,kSq,h,sigma):
	a = 1/(1 + 1j*np.sum(dW)*kSq);
	b = 1j*h/(1 + 1j*np.sum(dW)*kSq);
	nextU = np.multiply(a,currU) + np.multiply(b,fft(sf.cubicU(ifft(currU),sigma)))
	return nextU

def PSCN(currU,dW,kSq,h,sigma):
	nextU = sf.PSEulTypeSolver(currU,sum(dW),kSq,h,sigma,sf.CNNonLin)
	return nextU
	
def PSMEul(currU,dW,kSq,h,sigma):
	nextU = sf.PSEulTypeSolver(currU,sum(dW),kSq,h,sigma,sf.MEulNonLin)
	return nextU
	
def PSMP(currU,dW,kSq,h,sigma):
	nextU = sf.PSEulTypeSolver(currU,sum(dW),kSq,h,sigma,sf.MPNonLin)
	return nextU
	
def PSStrangSpl(currU,dW,kSq,h,sigma):
	# Half derivative step
	tempPhysCurrU = ifft(np.multiply(np.exp(-dW[0]*1j*kSq),currU))
	# Full nonlinear step
	tempU = fft(np.multiply(np.exp(h*1j*np.power(np.abs(tempPhysCurrU),2*sigma)),tempPhysCurrU))
	# Half derivative step
	nextU = np.multiply(np.exp(-dW[1]*1j*kSq),tempU)
	return nextU
	
def PSLieSpl(currU,dW,kSq,h,sigma):
	# Full derivative step
	tempPhysCurrU = ifft(np.multiply(np.exp(-sum(dW)*1j*kSq),currU))
	# Full nonlinear step
	nextU = fft(np.multiply(np.exp(h*1j*np.power(np.abs(tempPhysCurrU),2*sigma)),tempPhysCurrU))
	return nextU
	
def PSFourSpl(currU,dW,kSq,h,sigma):
	# Full nonlinear step
	tempPhysCurrU = ifft(currU)
	tempU = fft(np.multiply(np.exp(h*1j*np.power(np.abs(tempPhysCurrU),2*sigma)),tempPhysCurrU))
	# Full derivative step
	nextU = np.multiply(np.exp(-sum(dW)*1j*kSq),tempU)
	return nextU
	
def PSExplExp(currU,dW,kSq,h,sigma):
	nextU = np.multiply(np.exp(-sum(dW)*1j*kSq),currU) + 1j*h*np.multiply(np.exp(-sum(dW)*1j*kSq),fft(sf.cubicU(ifft(currU),sigma)))
	return nextU
	
def PSSymExp(currU,dW,kSq,h,sigma):
	NStar = sf.PSNStarSolver(currU,dW[0],kSq,h,sigma)
	nextU = np.multiply(np.exp(-sum(dW)*1j*kSq),currU) + np.multiply(h*np.exp(-dW[1]*1j*kSq),NStar)
	return nextU
	
###########################################################################################
def FDFEul(currU,dW,FDMatSq,h,sigma):
	A = sparse.eye(len(currU)) + 1j*sum(dW)*FDMatSq
	b = 1j*h
	nextU = np.matmul(A,currU) + b*cubicU(currU,sigma);
	return nextU

def FDBEul(currU,dW,FDMatSq,h,sigma):
	B = sparse.eye(len(currU)) - 1j*sum(dW)*FDMatSq
	b = 1j*h
	nextU = linalg.solve(B,currU + b*cubicU(currU,sigma))
	return nextU

def FDCN(currU,dW,FDMatSq,h,sigma):
	nextU = sf.FDEulTypeSolver(currU,sum(dW),FDMatSq,h,sigma,sf.CNNonLin)
	return nextU
	
def FDMEul(currU,dW,FDMatSq,h,sigma):
	nextU = sf.FDEulTypeSolver(currU,sum(dW),FDMatSq,h,sigma,sf.MEulNonLin)
	return nextU
	
def PSMP(currU,dW,FDMatSq,h,sigma):
	nextU = sf.FDEulTypeSolver(currU,sum(dW),FDMatSq,h,sigma,sf.MPNonLin)
	return nextU
	
def FDStrangSpl(currU,dW,FDMatSq,h,sigma):
	# Half derivative step
	temp = np.dot(linalg.expm(1j*dW[0]*FDMatSq),currU)
	# Full nonlinear step
	temp = h*1j*np.multiply(np.power(np.abs(temp),2*sigma),temp)
	# Half derivative step
	nextU = np.dot(linalg.expm(1j*dW[1]*FDMatSq),temp)
	return nextU
	
def FDLieSpl(currU,dW,FDMatSq,h,sigma):
	# Full derivative step
	temp = np.dot(linalg.expm(1j*sum(dW)*FDMatSq),currU)
	# Full nonlinear step
	nextU = h*1j*np.multiply(np.power(np.abs(temp),2*sigma),temp)
	return nextU
	
def FDFourSpl(currU,dW,FDMatSq,h,sigma):
	# Full nonlinear step
	temp = h*1j*np.multiply(np.power(np.abs(temp),2*sigma),currU)
	# Full derivative step
	nextU = np.dot(linalg.expm(1j*sum(dW)*FDMatSq),currU)
	return nextU
	
def FDExplExp(currU,dW,FDMatSq,h,sigma):
	nextU = np.dot(linalg.expm(1j*sum(dW)*FDMatSq), currU + 1j*h*cubicU(currU,sigma));
	return nextU
	
def FDSymExp(currU,dW,FDMatSq,h,sigma):
	NStar = FDNStarSolver(currU,dW[0],FDMatSq,h,sigma)
	nextU = np.matmul(linalg.expm(1j*sum(dW)*FDMatSq),currU) + h*np.matmul(linalg.expm(1j*dW[1]*FDMatSq),NStar);
	return nextU