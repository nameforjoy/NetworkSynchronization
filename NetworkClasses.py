"""
NetworkClasses
@author: JoyClimaco
"""

import numpy as np
from scipy.integrate import odeint
from odeintw import odeintw

class StuartLandau:
    
    """ Class defining a Stuart-Landau network object. 
        Holds the most important network properties of a network formed by N Stuart-Landau 
        oscillators, as well as its corresponding ODE system and time integration functions.
        
        Attributes
        ----------
        w : ndarray of shape (N)
            Natural frequency of each oscillator
        A : matrix (nparray) of shape (N,N)
            Adjacency matrix
        K : float
            Coupling parameter
        alpha: float
            Stuart-Landau limit cycle parameter
        F : ndarray of shape (N)
            Magnitude of external force for each oscillator
        Omega : float
            Frequency of the external force
        systype : string ('complex', 'rectangular', 'polar')
            Coordinates used in the Stuart-Landau ODE system
    """
    
    def __init__(self, w, A, K, alpha, F=None, Omega=0, systype='rectangular'):
        
        self.type = 'StuartLandau'                     # Network type
        self.w = w                                     # Natural frequencies
        self.N = np.size(w)                            # Number of oscillators
        self.A = A                                     # Adjacency matrix
        self.alpha = alpha                             # SL limit cycle parameter
        self.K = K                                     # Coupling constant
        self.edges = np.transpose(np.nonzero(A))       # Edges indexes
        self.Ne = np.size(self.edges,0)                # Nunber of edges
        self.F = F                                     # Force strength
        self.Omega = Omega                             # Frenquency of force
        self.forced_osc = np.nonzero(F)                # Forced oscillators' indexes
        self.f = np.size(self.forced_osc)/self.N       # Fraction of forced oscillators
        self.systype = systype                         # System type
        
    def __call__(self, z, t=0):
        
        """ Makes the StuartLandau object a callable function corresponding to the ODE system
        for the oscillators' positions z at time t
        """
        
        # Complex form of SL system
        if self.systype =='complex':
            
            # Isolated oscillator terms
            zdot = np.zeros(self.N, dtype='complex')
            for i in range(self.N):
                zdot[i] = (1j*self.w[i] + self.alpha**2 - z[i]*np.conj(z[i])) * z[i]
            
            # Forced terms
            if self.F is not None:
                for i in self.forced_osc:
                    zdot[i] += self.F[i] * np.exp(1j * self.Omega * t)
        
            # Coupling
            for k in range(self.Ne):
                [i, j] = self.edges[k]
                zdot[i] += self.K * self.A[i,j] * (z[j] - z[i])
                
            return zdot
        
        # SL system in polar coordinates
        if self.systype =='polar':
            
            z = np.reshape(z, (2,self.N))
            rho = z[0]
            theta = z[1]
            rhodot = np.zeros(self.N)
            thetadot = np.zeros(self.N)
        
            for i in range(self.N):
                rhodot[i] = (self.alpha**2 - rho[i]**2)*rho[i]
                thetadot[i] = self.w[i]
                
            if self.F is not None:
                for i in self.forced_osc:
                    rhodot[i] += self.F[i] * np.cos(t*self.Omega - theta[i])
                    thetadot[i] += self.F[i] * np.sin(t*self.Omega - theta[i])
            
            for k in range(self.Ne):
                [i, j] = self.edges[k]
                rhodot[i] += self.K * self.A[i,j] * (rho[j] * np.cos(theta[j] - theta[i]) - rho[i])
                thetadot[i] += self.K * self.A[i,j] * (rho[j]/rho[i]) * np.sin(theta[j] - theta[i])
                
            return np.concatenate((rhodot, thetadot))
            
        # SL system in rectangular coordinates
        if self.systype =='rectangular':

            z = np.reshape(z, (2,self.N))
            x = z[0]
            y = z[1]
            xdot = np.zeros(self.N)
            ydot = np.zeros(self.N)
        
            for i in range(self.N):
                aux = self.alpha**2 - x[i]**2 - y[i]**2
                xdot[i] = aux*x[i] - self.w[i]*y[i]
                ydot[i] = aux*y[i] + self.w[i]*x[i]
                
            if self.F is not None:
                for i in self.forced_osc:
                    xdot[i] += self.F[i] * np.cos(t*self.Omega)
                    ydot[i] += self.F[i] * np.sin(t*self.Omega)
            
            for k in range(self.Ne):
                [i, j] = self.edges[k]
                xdot[i] += self.K * self.A[i,j] * (x[j] - x[i])
                ydot[i] += self.K * self.A[i,j] * (y[j] - y[i])

            return np.concatenate((xdot, ydot))
    
    def integrate(self, z0, t0, tf=None, dt=0.1):
        
        """ Class method to calculate the system's time evolution through numerical integration
        
        Parameters:
        ----------
        z0 : complex ndarray of shape (N)
            State of system at initial time t0
        t0 : float or ndarray
            Initial time (float) or time array on which to evolve the system
        tf : float
            Final time value
        dt : float
            Size of steps in the time interval
        
        Returns
        ----------
        z : ndarray of shape (N, Nt)
            z[i] gives time evolution array for the ith oscillator
        t : ndarray of shape (Nt)
            Time values for which we evaluated the integral
        """
        
        if tf==None:
            t = t0
        else:
            t = np.arange(t0,tf,dt)
        
        # If the system is in its compex form, we use the special numerical integration
        # function odeintw, which extends scipy's odeint to accept complex ODE's
        # Obs: the initial condition z0 must be of type complex for this solver to work
        if self.systype =='complex':
            
            z = odeintw(self, z0, t).T
            
        if self.systype =='polar':
            
            # Jacobian for polar coordinates
            def Jac(z, t=0):
                
                z = np.reshape(z, (2,self.N))
                rho = z[0]
                theta = z[1]
                jac = np.zeros((2*self.N, 2*self.N))
                
                # Diagonal terms
                for i in range(self.N):
                    jac[i,i] = self.alpha**2 - 3 * rho[i]**2
                
                # Forced terms (off-diagonal)
                if self.F is not None:
                    for i in self.forced_osc:
                        jac[i,(i+self.N)] += - self.F[i] * np.sin(t*self.Omega - theta[i])
                        jac[(i+self.N),(i+self.N)] += self.F[i] * np.cos(t*self.Omega - theta[i])
                
                # Coupling terms (off-diagonal when assuming there are no self-edges)
                for k in range(self.Ne):
                    [i, j] = self.edges[k]
                    jac[i, j] += self.K * self.A[i,j] * np.cos(theta[j] - theta[i])
                    jac[i,(j+self.N)] = - self.K * self.A[i,j] * rho[j] * np.sin(theta[j] - theta[i])
                    jac[(i+self.N),j] = (self.K/rho[i]) * self.A[i,j] * np.sin(theta[j] - theta[i])
                    jac[(i+self.N),(j+self.N)] = self.K * self.A[i,j] * (rho[j]/rho[i]) * np.cos(theta[j] - theta[i])
                
                return jac
               
            rho0 = np.absolute(z0)
            theta0 = np.angle(z0)
            z0 = np.concatenate((rho0,theta0), axis=0)

            z = odeint(self, z0, t, Dfun=Jac).T
            rho = z[:self.N]
            theta = z[self.N:]
            z = rho*np.exp(1j*theta)
            
        if self.systype =='rectangular':
            
            # Jacobian for rectangular coordinates
            def Jac(z, t=0):
                
                z = np.reshape(z, (2,self.N))
                x = z[0]
                y = z[1]
                jac = np.zeros((2*self.N, 2*self.N))
                
                # Individual oscillator dynamics
                for i in range(self.N):
                    jac[i,i] = self.alpha**2 - y[i]**2 - 3*x[i]**2
                    jac[i,(i+self.N)] = - 2*x[i]*y[i] - self.w[i]
                    jac[(i+self.N),i] = - 2*x[i]*y[i] + self.w[i]
                    jac[(i+self.N),(i+self.N)] = self.alpha**2 - x[i]**2 - 3*y[i]**2
                  
                # Coupling terms
                for k in range(self.Ne):
                    [i,j] = self.edges[k]
                    jac[i,j] += self.K*self.A[i,j]
                    jac[i+self.N,j+self.N] += self.K*self.A[i,j]
                    
                return jac
            
            x0 = np.real(z0)
            y0 = np.imag(z0)
            z0 = np.concatenate((x0,y0), axis=0)
            
            z = odeint(self, z0, t, Dfun=Jac).T
            x = z[:self.N]
            y = z[self.N:]
            z = x + 1j*y
            
        return z, t
    
class KuramotoNetwork:
    
    """ Class defining a Kuramoto network object. 
        Holds the most important properties of a network formed by N Kuramoto oscillators, 
        as well as its corresponding ODE system and time integration functions.
        
        Attributes
        ----------
        w : ndarray of shape (N)
            Natural frequency of each oscillator
        A : matrix (nparray) of shape (N,N)
            Adjacency matrix
        K : float
            Coupling parameter
        F : ndarray of shape (N)
            Magnitude of external force for each oscillator
        Omega : float
            Frequency of the external force
    """
    
    def __init__(self, w, A, K, F=None, Omega=0):
        
        self.type = 'Kuramoto'                         # Network type
        self.w = w                                     # Natural frequencies
        self.N = np.size(w)                            # Number of oscillators
        self.A = A                                     # Adjacency matrix
        self.K = K                                     # Coupling constant
        self.edges = np.transpose(np.nonzero(A))       # Edges indexes
        self.Ne = np.size(self.edges,0)                # Nunber of edges
        self.F = F                                     # Force strength
        self.Omega = Omega                             # Frenquency of force
        self.forced_osc = np.nonzero(F)                # Forced oscillators' indexes
        self.f = np.size(self.forced_osc)/self.N       # Fraction of forced oscillators


    def __call__(self, theta, t=0):
        
        """ Makes the StuartLandau object a callable function corresponding to the ODE system
        for the oscillators' phases theta at time t
        """
        
        thetadot = np.zeros(self.N)
        
        # Individual oscillator dynamics
        for i in range(self.N):     
            thetadot[i] += self.w[i]
        
        # Coupling terms
        for k in range(self.Ne): # loops only through the nonzero elements of A
            [i, j] = self.edges[k]
            thetadot[i] += self.K * self.A[i, j] * np.sin(theta[j] - theta[i])
        
        # Forced terms (if there are any)
        if self.F is not None:
            for i in self.forced_osc:
                thetadot[i] += self.F[i] * np.sin(t*self.Omega - theta[i])
                
        return thetadot
    
    def integrate(self, theta0, t0, tf=None, dt=0.1):
        
        """ Class method to calculate the system's time evolution through numerical integration
        
        Parameters:
        ----------
        theta0 : ndarray of shape (N)
            State of system at initial time t0
        t0 : float or ndarray
            Initial time (float) or time array on which to evolve the system
        tf : float
            Final time value
        dt : float
            Size of steps in the time interval
        
        Returns
        ----------
        z : ndarray of shape (N, Nt)
            z[i] gives time evolution array for the ith oscillator
        t : ndarray of shape (Nt)
            Time values for which we evaluated the integral
        """
        
        if tf==None:
            t = t0
        else:
            t = np.arange(t0,tf,dt)
            
        # Jacobian for the ODE system
        def Jac(theta, t=0):
            
            jac = np.zeros((self.N, self.N))
            for k in range(self.Ne):
                [i, j] = self.edges[k]
                jac[i, j] = self.K * self.A[i,j] * np.cos(theta[j] - theta[i])
                
            if self.F is not None:
                for i in self.forced_osc:
                    jac[i,i] += - self.F[i] * np.cos(t*self.Omega - theta[i])
            
            return jac
            
        theta = odeint(self, theta0, t, Dfun=Jac).T
        
        return theta, t