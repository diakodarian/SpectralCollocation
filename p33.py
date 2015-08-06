# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:00:17 2015

@author: Diako Darian
"""

# p33.py - solve u_xx = exp(4x), u_x(-1)=g'minus, u(1)=gplus
# This code solves the above equation for mixed boundary conditions 

from numpy import *
from pylab import *
from cheb  import cheb

N = 16; D,x = cheb(N);
A = D[:,0:N]; B = D[0:N,:]; L = dot(A,B); 
v = D[0:N+2,N];

f = exp(4*x[1:N+2]);
  
# Boundary conditions
gminus = 0.0;
gplus = 0.0;
  
gminus_prime = 0.0;
gplus_prime = 0.0;
  
# Impose Neumann BC:
ff = f-gminus_prime*v[1:N+2];
  
# Impose Dirichlet BC:
F = ff-L[1:N+2,0]*gplus;
#F = F - (L(2:N+1,end))*gminus;
  
sol = linalg.solve(L[1:N+2,1:N+2],F);     
u = hstack(([gplus], sol));              
  
plot(x,u,'.',markersize=8)
xx = arange(-1,1.01,.01)
uu = polyval(polyfit(x,u,N),xx)    # interpolate grid data
plot(xx,uu)
grid(); xlim(-1,1)
exact =  (exp(4*xx) +(16*gminus_prime - 4*exp(-4))*(xx-1) - exp(4)+16*gplus)/16
title('max err = '+str('%.3e' % norm(uu-exact,inf)),fontsize=12)
