# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:20:21 2015

@author: Diako Darian
"""

# p34.py - Allen-Cahn eq. u_t = eps*u_xx+u-u^3, u(-1)=-1, u(1)=1
#         (compare p6.py and p32.py)

from numpy import *
from pylab import *
from cheb  import cheb
from mpl_toolkits.mplot3d import Axes3D

# Differentiation matrix and initial data:
N = 20; D,x = cheb(N); D2 = dot(D,D);     # use full-size matrix
D2[(0,N),:] = zeros((2,N+1));                # for convenience
eps = 0.01; dt = min([.01,50*N**(-4)/eps]);
t = 0; v = .53*x + .47*sin(-1.5*pi*x);
gplus = 1; gminus = -1;

# Solve PDE by Euler formula and plot results:
tmax = 100; tplot = 2; nplots = int(round(tmax/tplot));
plotgap = int(round(tplot/dt)); #dt = floor(tplot/plotgap);
xx = arange(-1,1.01,.01); vv = polyval(polyfit(x,v,N),xx);  # interpolate grid data

plotdata = vstack(( vv, zeros((nplots,201)) )); tdata = [0.]

for i in range(1,nplots):
    for n in range(1,plotgap):
        t = t+dt

        v[1:N] = v[1:N] + dt*(eps*(D2[1:N,0]*gplus + dot(D2[1:N,1:N],v[1:N])+ D2[1:N,N]*gminus) + v[1:N] - v[1:N]**3);    # Euler
        
    vv = polyval(polyfit(x,v,N),xx);
    plotdata[i,:] = vv; tdata.append(dt*i*plotgap)

fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y = meshgrid(xx, tdata)
ax.plot_surface(X, Y, plotdata[:nplots,:], cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u")
plt.show()

