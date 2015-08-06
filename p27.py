# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 08:06:48 2015

@author: Diako Darian
"""

# p27.py - Solve KdV eq. u_t + uu_x + u_xxx = 0 on [-pi,pi] by
#         FFT with integrating factor v = exp(-ik^3t)*u-hat.

from numpy import *
from waterfall import waterfall
from numpy.fft import fft,ifft

def sech(x):
    return cosh(x)**(-1)
    
# Set up grid and two-soliton initial data:
N = 256; dt = .3/N**2; x = (2*pi/N)*arange(-N/2,N/2);
A = 25; B = 16; 
u = 3*A**2*sech(.5*(A*(x+2)))**2 + 3*B**2*sech(.5*(B*(x+1)))**2; 
v = fft(u); 
k = hstack((arange(0,N/2),[0.],arange(-N/2+1,0))); ik3 = 1j*k**3; 

# plot parameters
tmax = 0.006; nplt = floor((tmax/25)/dt); nmax = int(round(tmax/dt)); 
t = 0.

plotdata = vstack(( u, zeros((25,N)) )); tdata = [0.]; n = 0

# RK4 parameters:
a = array([1./6, 1./3, 1./3, 1./6])*dt;
b = array([.5, .5, 1])*dt;

def computeRHS(t,v):
    return -.5j*exp(-t*ik3)*k*fft((real(ifft(exp(ik3*t)*v)))**2)
    
for i in range(1,nmax):
    t = i*dt
    v1 = v.copy(); v0 = v.copy();
    for rk in range(4):
        du = computeRHS(t,v);
        if rk < 3:
            v = v0; v = v + b[rk]*du;
        v1 = v1 + a[rk]*du;
    v = v1; 
    
    if i%nplt == 0: 
        n += 1
        plotdata[n,:] = real(ifft(exp(ik3*t)*v)); 
        tdata.append(t)

waterfall(x,tdata,plotdata,labels=['x','t','u'],slabthickness=0.2,zrange=[0,2000.])
show()