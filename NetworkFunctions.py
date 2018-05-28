"""
NetworkFunctions
@author: JoyClimaco
"""

import numpy as np

def OrderParameter(network, state0, t0, tf, K0, Kf=None, dK=0.1, dt=0.1, output='all'):
    
    """ Calculates the order parameter (as well as other synchronization parameters) 
    of the given network for one or seveal coupling constants, averaged in the given time interval 
    
    Parameters
    ----------
    network : StuartLandau or KuramotoNetwork object
        Network of N oscillators
    state0 : ndarray of shape (N)
        Network's initial state
    t0 : float
        Beggining of the time interval on which we evaluate the parameters
    tf : float
        End of the time interval
    K0 : float
        Initial network coupling constant
    Kf : float
        Final coupling constant on which we evaluate the parameters
    dK : float
        Size of the steps inside the coupling constants' interval
    dt : float
        Size of the steps inside the time interval
    output: 'all', 'simple', 'rpsi'
        Constrols the output type, and which parameters will be given
        Obs: unless chosen as 'all', the function will not calculate phidot
        
    Returns
    -------
    K : ndarray of shape (NK)
        Coupling constants for which the parameters are evaluated
    r : ndarray of shape (NK)
        Order parameter (averaged in the given time interval for each K)
    r_std : ndarray of shape (NK)
        Standard deviation for each averaged order parameter
    psi : ndarray of shape (NK)
        Mean phase (averaged in the given time interval for each K)
    psi_std : ndarray of shape (NK)
        Standard deviation for each averaged mean phase
    psidot : ndarray of shape (NK)
        Average angular speed of the mean phase for the given time interval
    psidot_std : ndarray of shape (NK)
        Standard deviation of the averaged mean phase's angular speed
    """
    
    if Kf==None:
        K = np.array([K0])
    else:
        K = np.arange(K0,Kf,dK)
    NK = np.size(K)
           
    t = np.arange(0,tf,dt)   
    Nt = np.size(np.arange(t0,tf,dt))
    Nt0 = np.size(t) - Nt
    
    r = np.zeros(NK)
    r_std = np.zeros(NK)
    psi = np.zeros(NK)
    psi_std = np.zeros(NK)
    psidot = np.zeros(NK)
    psidot_std = np.zeros(NK)
    
    for j in range(NK):
        
        network.K = K[j]
        
        if network.type=='StuartLandau':
            z, _ = network.integrate(state0, t)
            theta = np.transpose(np.angle(z))
            order_par = np.zeros(Nt, dtype='complex')
            
        if network.type=='Kuramoto':
            theta, _ = network.integrate(state0, t)
    
        for i in range(Nt):
            order_par[i] = np.sum(np.exp(theta[i+Nt0]*1j), dtype='complex')/network.N
            # mean value over all oscillators for all values of time in the given interval
    
        aux = np.absolute(order_par)
        r[j] = np.average(aux)
        r_std[j] = np.std(aux)

        aux = np.angle(order_par)
        psi[j] = np.average(aux)
        psi_std[j] = np.std(aux)
        
        if output=='all':
            vel = np.zeros(Nt-1)
            for i in range(Nt-1):
                vel[i] = (aux[i+1] - aux[i])/dt
            
            psidot[j] = np.average(vel)
            psidot_std[j] = np.std(vel)
        
    if NK==1:
        K = K[0]
        r = r[0]
        r_std= r_std[0]
        psi = psi[0]
        psi_std = psi_std[0]
        psidot = psidot[0]
        psidot_std = psidot_std[0]
        
    if output=='simple':
        return r, psi, psidot
    
    if output=='rpsi':
        sync_par = {'K': K,
                     'r': r,
                     'r_std': r_std,
                     'psi': psi,
                     'psi_std': psi_std
                     } 
        return sync_par
    
    if output=='all':
        sync_par = {'K': K,
                     'r': r,
                     'r_std': r_std,
                     'psi': psi,
                     'psi_std': psi_std,
                     'psidot': psidot,
                     'psidot_std': psidot_std
                     }        
        return sync_par
    
def AverageOrderPar(network, N0, t0, tf, K0, Kf=3, dK=0.2, dt=0.1, output='all'):
    
    """ Evolves the previously defined function OrderParameter for N0 different initial states
    and takes the average and standard deviations of the output synchronization parameters in
    the given set of N0 initial conditions.
    
    We define the final standard deviation as the average value of average(r+r_std) - average(r)
    for each initial state
    """

    NK = np.size(np.arange(K0,Kf,dK))
    r = np.zeros((N0, NK))
    r_std = np.zeros((N0, NK))
    psi = np.zeros((N0, NK))
    psi_std = np.zeros((N0, NK))
    psidot = np.zeros((N0, NK))
    psidot_std = np.zeros((N0, NK))

    for i in range(N0):
        
        # Creates Stuart-Landau random initial state
        if network.type=='StuartLandau':
            rho0 = np.random.uniform(0.1, 0.9, network.N)
            theta0 = np.random.uniform(0, 2*np.pi, network.N)
            state0 = rho0 * np.exp(1j * theta0)
        
        # Creates Kuramoto random initial state
        if network.type=='Kuramoto':
            state0 = np.random.uniform(0, 2*np.pi, network.N)
            
        # Evolves synchronization parameters for the initial conditions created above
        sync_par = OrderParameter(network, state0, t0, tf, K0, Kf=Kf, dK=dK, dt=dt, output=output)
        
        r[i] = sync_par['r']
        r_std[i] = sync_par['r_std'] + r[i]
        psi[i] = sync_par['psi']
        psi_std[i] = sync_par['psi_std'] + psi[i]
        
        if output=='all':
            psidot[i] = sync_par['psidot']
            psidot_std[i] = sync_par['psidot_std'] + psidot[i]
        
    K = sync_par['K']
    
    r = np.average(r, axis=0)
    psi = np.average(psi, axis=0)

    r_std = np.average(r_std, axis=0) - r
    psi_std = np.average(psi_std, axis=0) - psi

    if output=='rpsi':
        sync_par = {'K': K,
                    'r': r,
                    'r_std': r_std,
                    'psi': psi,
                    'psi_std': psi_std}

    if output=='all':
        
        psidot = np.average(psidot, axis=0)
        psidot_std = np.average(psidot_std, axis=0) - r

        sync_par = {'K': K,
                     'r': r,
                     'r_std': r_std,
                     'psi': psi,
                     'psi_std': psi_std,
                     'psidot': psidot,
                     'psidot_std': psidot_std}
        
    return sync_par
    
def GammaCoupling(network, shape, scale=1, direc=False):
    
    """ Calculates the order parameter (as well as other synchronization parameters) 
    of the given network for one or seveal coupling constants, averaged in the given time interval 
    
    Parameters
    ----------
    network : StuartLandau or KuramotoNetwork object
        Initial network
    shape : float
        Shape paameter of the gamma distribution
    scale : float
        Scale parameter of the gamma distribution
    direc : bool
        False if the output network is undirected, True otherwise
        
    Returns
    -------
    network : StuartLandau or KuramotoNetwork object
        Initial network with gamma distribuited edge weights
    Kmean : float
        Average value of the gamma distribuition
    Kstd : float
        Standard deviation of the gamma distribution
    """
    
    n = network.Ne
    nz = network.edges
    network.weight = 'gamma'
    
    # Takes away the edges with i>j for the undirected case
    if direc==False:
        aux = []
        for k in range(n):
            [i, j] = nz[k]
            if i<j:
                aux.append([i,j])
        nz = np.array(aux)
        n = np.size(nz,0)
    
    # Creates gamma distribution
    Karray = np.random.gamma(shape, scale=scale, size=n)
    Kmean = np.mean(Karray)
    Kstd = np.std(Karray)
    
    for k in range(n):
        [i, j] = nz[k]
        network.A[i,j] = Karray[k]/Kmean # normalizing
        
        # Assures A is symmetric for the undirected case
        if direc==False:
            network.A[j,i] = Karray[k]/Kmean
    
    return network, Kmean, Kstd