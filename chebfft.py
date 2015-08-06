# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:27:22 2015

@author: Diako Darian
"""

# CHEBFFT  Chebyshev differentiation via FFT. Simple, not optimal.  
#          If v is complex, delete "real" commands.
from numpy import *
from numpy.fft import fft,ifft
def chebfft(v):
	N = len(v)-1; 
	if N==0: return 0
	x = cos(arange(0,N+1)*pi/N)
	ii = arange(0,N); iir = arange(1-N,0); iii = array(ii,dtype=int)
	#v = v[:]
	V = hstack((v,v[N-1:0:-1])) # transform x -> theta 
	U = real(fft(V))
	W = real(ifft(1j*hstack((ii,[0.],iir))*U))
	w = zeros(N+1)
	w[1:N] = -W[1:N]/sqrt(1-x[1:N]**2)    # transform theta -> x     
	w[0] = sum(iii**2*U[iii])/N + .5*N*U[N]     
	w[N] = sum((-1)**(iii+1)*ii**2*U[iii])/N + .5*(-1)**(N+1)*N*U[N]
	return w









#print chebfft(array([1,2,3.,4.,6.,5.,-2.1])).reshape(7,1)