#! /usr/bin/env python
from scipy.fftpack import fft, ifft
import numpy as np
#	IMPLICITSOLVING
#def CrankNic(currU,dW,k,h):
#def MidPoint(currU,dW,k,h):
#def MidEul(currU,dW,k,h):
#def SymExp(currU,dW,k,h):

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



	
#                    case 1 % Crank-Nicolson
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        if sigma == 1
#                            CNG = @(u,v) (abs(u).^2+abs(v).^2).*(u+v)/4;
#                        else
#                            CNG = @(u,v) (abs(u).^(2*(sigma+1))-abs(v).^(2*(sigma+1)))./...
#                                (abs(u).^2-abs(v).^2).*(u+v)/(2*(sigma+1));
#                        end
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            currU = myPSImplSolver(currU,k,sum(dW),currh,CNG);
#                        end
#                    case 2 % Midpoint scheme
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        MG = @(u,v) abs((u+v)/2).^(2*sigma).*(u+v)/2;
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            currU = myPSImplSolver(currU,k,sum(dW),currh,MG);
#                        end
#                    case 3 % Theta Euler scheme, theta=0.5, version of midpoint
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        MG = @(u,v) abs(u).^(2*sigma).*u;
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            currU = myPSImplSolver(currU,k,sum(dW),currh,MG);
#                        end
#                    case 4 % Theta Euler scheme, theta=1, Explicit Euler
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            a = (1 - 1i*sum(dW)*k.^2);
#                            b = 1i*currh;
#                            currU = a.*currU + b.*fft(cubicU(ifft(currU)));
#                        end
#                    case 5 % Theta Euler scheme, theta=0, Implicit Euler
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            a = 1./(1 + 1i*sum(dW)*k.^2);
#                            b = 1i*currh./(1 + 1i*sum(dW)*k.^2);
#                            currU = a.*currU + b.*fft(cubicU(ifft(currU)));
#                        end
#                    case 6 % Theta splitting scheme, theta=0.5, Strang splitting
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            % Half first step
#                            temp = exp(-dW(1)*1i*k.^2).*currU;
#                            % Full nonlinear step
#                            tempRealSpace = ifft(temp);
#                            temp = fft(exp(currh*1i*abs(tempRealSpace).^(2*sigma)).*tempRealSpace);
#                            % Half last step
#                            currU = exp(-dW(2)*1i*k.^2).*temp;
#                        end
#                    case 7 % Theta splitting scheme, theta=1, Lie/Fourier splitting
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            % Full first step
#                            tempRealSpace = ifft(exp(-sum(dW)*1i*k.^2).*currU);
#                            % Full nonlinear step
#                            currU = fft(exp(currh*1i*abs(tempRealSpace).^(2*sigma)).*tempRealSpace);
#                        end
#                    case 8 % Theta splitting scheme, theta=0, Lie/Fourier splitting
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            % Full nonlinear step
#                            tempRealSpace = ifft(currU);
#                           temp = fft(exp(currh*1i*abs(tempRealSpace).^(2*sigma)).*tempRealSpace);
#                            % Full first step
#                            currU = exp(-sum(dW)*1i*k.^2).*temp;
#                        end
#                    case 9 % Explicit exponential scheme
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            % Full first step
#                            currU = exp(-sum(dW)*1i*k.^2).*currU + 1i*currh*exp(-sum(dW)*1i*k.^2).*fft(cubicU(ifft(currU)));
#                        end
#                    case 10 % Symmetric exponential scheme
#                        % Initialization
#                        currU = fft(u0FunVal);
#                        for i = 1:currN
#                            dW = W(i,:);
#                            %% Scheme calculations
#                            NStar = myPSNStarSolver(currU,k,dW(1),currh,sigma);
#                            currU = exp(-sum(dW)*1i*k.^2).*currU + currh*exp(-dW(2)*1i*k.^2).*NStar;
#                        end
#                end