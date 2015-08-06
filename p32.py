# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:33:57 2015

@author: Diako Darian
"""

# p32.py - solve u_xx = exp(4x), u(-1)=0, u(1)=1 (compare p13.py)

from numpy import *
from pylab import *
from cheb  import cheb

N = 16;
D,x = cheb(N); D2 = dot(D,D); D2M = D2[1:N,1:N];
f = exp(4*x[1:N]);
  
# Boundary conditions
gminus = 0.0;
gplus = 1.0;
  
F = f-D2[1:N,0]*gplus;
F = F - (D2[1:N,-1])*gminus;
           
v = linalg.solve(D2M,f)             # Poisson eq. solved here
v = hstack(([0.],v,[0.])) + (x+1.)/2. 

u = linalg.solve(D2M,F)             # Poisson eq. solved here
u = hstack(([gplus],u,[gminus]))

plot(x,u,'.',markersize=8)
xx = arange(-1,1.01,.01)
uu = polyval(polyfit(x,u,N),xx)    # interpolate grid data
plot(xx,uu)
grid(); xlim(-1,1)
exact = ( exp(4*xx) - sinh(4)*xx - cosh(4) )/16 + (xx+1)/2
title('max err = '+str('%.3e' % norm(uu-exact,inf)),fontsize=12)
figure()
plot(x,v,'.',markersize=8)
xx = arange(-1,1.01,.01)
uu = polyval(polyfit(x,v,N),xx)    # interpolate grid data
plot(xx,uu)
grid(); xlim(-1,1)
exact = ( exp(4*xx) - sinh(4)*xx - cosh(4) )/16 + (xx+1)/2
title('max err = '+str('%.3e' % norm(uu-exact,inf)),fontsize=12)
show()
