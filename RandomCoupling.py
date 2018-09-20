""""
@author: JoyClimaco
"""
import numpy as np
import scipy.stats as ss
import networkx as nx
from NetworkFunctions import RandomCoupling
from NetworkFunctions import OrderParameter
from NetworkClasses import StuartLandau

A = np.load('A_BA_m2_N200_1.npy') # load adjacency matrix
w = np.load('w200_3unif.npy') # load frequencies

N = np.size(A,0) # network size
K = .5 # coupling constant
alpha = 1 # SL parameter

# initial conditions
theta0 = np.random.uniform(0, 2*np.pi, N)
rho0 = np.random.uniform(0.1, 0.9, N) # so the system doesn't fall into the attractor
z0 = rho0*np.exp(1j*theta0)

# Defines Stuart-Landau system
SL = StuartLandau(w, A, K, alpha)

# Random array for the coupling constants
Karray = np.random.gamma(shape=2, scale=1, size=SL.Ne)
np.save('z_Karray.npy', Karray)

# Defines new SL system with this coupling weights
SL_rand = RandomCoupling(SL, Karray, dist_type='Gamma', shape=2, scale=.5)

# Time evolution of the oscillators
t = np.arange(0,50,.2)
z, _ = SL_rand.integrate(z0, t)
np.save('z_time.npy', t)
np.save('z_evolution.npy', z)

# Order parameter calculation
K, r, r_std = OrderParameter(SL_rand, z0, 30, 35, .05, Kf=3, dK=.05, dt=.1, output='simple')

np.save('z_K.npy', K)
np.save('z_r.npy', r)
np.save('z_r_std.npy', r_std)