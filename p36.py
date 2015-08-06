# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:41:47 2015

@author: Diako Darian
"""

# p36.py - Laplace eq. on [-1,1]x[-1,1] with nonzero BCs

from numpy import *
from pylab import *
from cheb  import cheb
from mpl_toolkits.mplot3d import Axes3D

# Set up grid and 2D Laplacian, boundary points included:
N = 24; D,x = cheb(N); y = x.copy();
X,Y = meshgrid(x,y);
xx = X.reshape((N+1)*(N+1)); yy = Y.reshape((N+1)*(N+1)) # stretch 2D grids to 1D vectors
                        # Transpose because default storage in numpy arrays is by rows.
     
D2 = dot(D,D); I = eye(N+1); L = kron(I,D2) + kron(D2,I);     

# Impose boundary conditions by replacing appropriate rows of L:
b = (abs(xx)==1) + (abs(yy)==1);            # boundary pts
L[b,:] = zeros((4*N,(N+1)*(N+1)))
L[b,b] = 1;
rhs = zeros(((N+1)**2)); 
rhs[b] = (yy[b]==1)*(xx[b]<0)*sin(pi*xx[b])**4 + .2*(xx[b]==1)*sin(3*pi*yy[b]);


# Solve Laplace equation, reshape to 2D, and plot:
u = linalg.solve(L,rhs); uu = u.reshape((N+1,N+1));
value = uu[N/2,N/2]
  
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, uu, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
ax.azim = -138; ax.elev = 25
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
ax.text(0,.8,.4,'u(0,0) = '+str('%.11f' % value) )

plt.show()