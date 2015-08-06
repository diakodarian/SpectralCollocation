# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:37:10 2015

@author: Diako Darian
"""

# p37.py - 2D "wave tank" with Neumann BCs for |y|=1

'''
We use Fourier discretization in x-direction and Chebyshev discretization in y
We use Dirichlet BC: u(-1)=u(1)=0
FFT is used in x and CHebyshev matrix in y
'''
from numpy import *
from pylab import *
from cheb  import cheb
from numpy.fft import fft,ifft
from mpl_toolkits.mplot3d import axes3d, Axes3D
from chebfft  import chebfft

# x variable in [-A,A], Fourier:
A = 3.; Nx = 50; dx = 2.*A/Nx; x = -A+dx*arange(1,Nx+1)
  
# y variable in [-1,1], Chebyshev:
Ny = 15; y = cos(pi*arange(Ny+1)/Ny)

# Wave number vector
k = hstack((arange(0,Nx/2),[0.],arange(-Nx/2+1,0))); k2 = -1*k*k

# Grid and initial data:
xx,yy = meshgrid(x,y);
vv = exp(-8*((xx+1.5)**2+yy**2));
dt = 5./(Nx+Ny**2); 
vvold = exp(-8*((xx+dt+1.5)**2+yy**2));
gplus = 0; gminus = 0;

# Time-stepping by leap frog formula:
plotgap = int(round(2./dt)); dt = 2./plotgap;
for n in range(2*plotgap+1):
    t = n*dt
    if (n+.5)%plotgap < 1.:
        i = n/plotgap
        ax = subplot(2,2,i+1,projection='3d')
        ax.plot_surface(xx, yy, vv, cstride=1,rstride=1,cmap='summer')
        title('t = '+str('%.5f' % t)); ax.azim = -138; ax.elev = 38; ax.set_zlim3d(-0.2,1)
        
    # Fourier transform in x-direction
    uxx = zeros((Ny+1,Nx)); uyy = zeros((Ny+1,Nx));
    for i in range(Ny):                # 2nd derivs wrt x in each row
        v = vv[i,:];
        v_hat = fft(v);
        w_hat = (pi/A)**2*k2*v_hat
        w = real(ifft(w_hat)); 
        uxx[i,:]= w;
    for j in range(Nx):                # 2nd derivs wrt y in each column
        v = vv[:,j]; 
        uyy[:,j] = chebfft(chebfft(v)); 
        # Impose Dirichlet BC:
        uyy[0,j]=gplus; uyy[-1,j]=gminus;
     
    vvnew = 2*vv - vvold + dt**2*(uxx+uyy);
    vvold = vv; vv = vvnew;

show()