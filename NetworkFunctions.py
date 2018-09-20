"""
NetworkFunctions
@author: JoyClimaco
"""

import numpy as np

def OrderParameter(network, state0, t0, tf, K0, Kf=None, dK=0.2, dt=0.1, output='all'):
    
    """ Calculates the order parameter (as well as other synchronization parameters) of the
    given network for one or seveal coupling constants, averaged in the given time interval 
    
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
        if type(K0)==int or type(K0)==float:
            K = np.array([K0])
        else:
            K = K0
    else:
        K = np.arange(K0,Kf,dK)
    NK = np.size(K)
           
    t = np.arange(0,tf,dt)   
    Nt = np.size(np.arange(t0,tf,dt))
    Nt0 = np.size(t) - Nt
    order_par = np.zeros(Nt, dtype='complex')
    
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
            
        if network.type=='Kuramoto':
            theta, _ = network.integrate(state0, t)
            theta = np.transpose(theta)
    
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
        return K, r, r_std
    
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
    
def AverageOrderPar(network, N0, t0, tf, K0, Kf=None, dK=0.2, dt=0.1, output='all'):
    
    """ Evolves the previously defined function OrderParameter for N0 different initial states
    and takes the average and standard deviations of the output synchronization parameters in
    the given set of N0 initial conditions.
    
    We define the final standard deviation as the average value of average(r+r_std) - average(r)
    for each initial state
    """
    
    if Kf==None:
        NK=1
        r = np.zeros(N0)
        r_std = np.zeros(N0)
        psi = np.zeros(N0)
        psi_std = np.zeros(N0)
        psidot = np.zeros(N0)
        psidot_std = np.zeros(N0)
    else:
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
        psidot_std = np.average(psidot_std, axis=0) - psidot

        sync_par = {'K': K,
                     'r': r,
                     'r_std': r_std,
                     'psi': psi,
                     'psi_std': psi_std,
                     'psidot': psidot,
                     'psidot_std': psidot_std}
        
    return sync_par

def FullSyncCoupling(network, K0, t0, tf, dt=0.5, N0=4, step0=1, r_f=.95, error_psidot=0.005, \
                     psidot_f=.01, error_r=.005, epsilon=.005, max_iter=30, av_half_N0=6, av_N0=3):
    
    """ Calculates the minimum coupling strength for which the system globally synchronizes
    
    Parameters
    ----------
    network : StuartLandau or KuramotoNetwork object
        Network of N oscillators
    K0 : float
        Initial guess for the full sync coupling
    t0 : float
        Beggining of the time interval on which we evaluate the order parameters
    tf : float
        End of the time interval
    K0 : float
        Initial network coupling constant
    dt : float
        Size of the steps inside the time interval
    N0 : int
        Maximmum number of initial conditions tests to test for each coupling value
    step0 : float
        First step (with sign) to give on K
    r_f : float
        Minimum value of the order parameter (r) for full synchronization
    error_psidot : float
        Maximum error on the mean phase velocity (psidot) for full sync
    psidot_f : float
        Maximum absolute value of psidot for full sync
    error_r : float
        Maximum error on r for full sync
    epsilon : float
        Precision of the final answer
    max_iter : int
        Maximum of iterations allowed
    av_half_N0 : float
        Defines the step size as a multiple of the error (step = av_half_N0*epsilon) 
        for which we start calculating r as the average for N0/2 initial states
    av_N0 : float
        Defines step size for which we start computing r for N0 initial states
        
    Returns
    -------
    K : float
        Full Synchronization Coupling
    crit_par : dict
        Dictionary with all calculated parameters for the critical K
    History : dict
        History of all coupling values tried and its respectives parameters
    """

    sgn = np.sign(step0) # step direction (negative for left, poditive for right)
    step = np.absolute(step0) # step magnitude
    K = K0 - sgn*step # so that the first loop calculation is for K0
    i = 0    
    Nmax = N0
    N0 = 1
        
    History = {'Kf_hist': [], 'rf_hist': [], 'psidotf_hist': [], 'r_stdf_hist': [], \
               'psidot_stdf_hist': [], 'step_hist': [], 'N0': []}
    
    while step > epsilon/2 and i < max_iter:
        
        i += 1 # updates iteration number
        K_ans = K
        K = K + sgn*step # Adds the current step to the coupling constant
        
        if K < 0:
            step = step + K
            K = epsilon
        
        if Nmax > 1:
            if step <= epsilon*(2**av_N0):
                N0 = Nmax
            else:
                if step <= epsilon*(2**av_half_N0) and Nmax > 3:
                    N0 = int(Nmax/2)
        
        # If this K was already tested, take the previously calculated values from History
        if K in History['Kf_hist']:
            j = History['Kf_hist'].index(K)
            
            r = History['rf_hist'][j]
            r_std = History['r_stdf_hist'][j]
            psidot = History['psidotf_hist'][j]
            psidot_std = History['psidot_stdf_hist'][j]
        
        # If this K is new, calculate the synchronization parameters
        else: 
            sync_par = AverageOrderPar(network, N0, t0, tf, K, dt=dt)
        
            r = sync_par['r']
            r_std = sync_par['r_std']
            psidot = sync_par['psidot']
            psidot_std = sync_par['psidot_std']
        
        # if any condition is not met, increase the coupling (sgn=1)
        if r < r_f or np.absolute(psidot) > psidot_f or \
        r_std > error_r and psidot_std > error_psidot:
            
            d = sgn # if the previous step direction is the same, d=1. Else, d=-1
            sgn = 1
        
        # if all conditions are met, try decreasing K so we may get its smallest value
        else:
            d = -sgn # if the previous move was to increase K, then d=-1. Else, d=1
            sgn = -1
            
        # if we switched the step direction (d=-1), cut the step size in half
        if d < 0:
            step = step/2
        
        # If it's the last step, choose between the last result and the previous one
        if step <= epsilon/2 or i==max_iter:
            
            # if the last step doesn't meet all the criteria but the previous one does, choose the latter
            if sgn > 0 and d < 0:
                K = K_ans
                r = History['rf_hist'][-1]
                psidot = History['psidotf_hist'][-1]
                r_std = History['r_stdf_hist'][-1]
                psidot_std = History['psidot_stdf_hist'][-1]
            
        # Save the parameters obtained in this step at the History dictionary
        History['Kf_hist'].append(K)
        History['rf_hist'].append(r)
        History['psidotf_hist'].append(psidot)
        History['r_stdf_hist'].append(r_std)
        History['psidot_stdf_hist'].append(psidot_std)
        History['step_hist'].append(step)
        History['N0'].append(N0)
        
    crit_par = {'Kf': K, 'rf': r, 'psidotf': psidot, 'r_stdf': r_std, 'psidot_stdf': psidot_std, 'iterations': i}
    
    return K, crit_par, History

def CriticalFraction(network, F, K, f0, t0, tf, dt=0.5, step0=.3, r_f=.95, N0=4, error_psidot=.005, \
                     error_r=.005, epsilon=.001, psidot_f=.01, max_iter=30, av_half_N0=6, av_N0=3):
    
    """ Calculates the minimum fraction of oscillators for which the system synchronizes with the force
    
    Parameters
    ----------
    network : StuartLandau or KuramotoNetwork object
        Network of N oscillators
    F : float
        Magnitude of force
    K : float
        Coupling constant
    f0 : float
        Initial guess for the critical fraction
        
    Returns
    -------
    fcrit : float
        Cirtical fraction
    crit_par : dict
        Dictionary with all calculated parameters for the critical fraction
    History : dict
        History of all coupling values tried and its respective parameters
    """

    network.rot_frame = True
    N = network.N # Total number of oscillators
    i = 0 
    Nmax = N0
    N0 = 1
    sgn = 1
    
    # Translates the fraction quantities into their corresponding number of oscillators
    epsilon = int(epsilon*N)
    n = int(f0*N) # starting number of forced oscillators
    step = int(step0*N)
    n = n - sgn*step

    History = {'nf': [], 'rf_hist': [], 'psidotf_hist': [], 'r_stdf_hist': [], \
               'psidot_stdf_hist': [], 'step_hist': [], 'N0': []}
    
    while step > epsilon/2 and i < max_iter:
        
        # For small steps increases the number of averages in the OrderParameter calculation
        if Nmax > 1:
            if step <= epsilon*(2**av_N0):
                N0 = Nmax
            else:
                if step <= epsilon*(2**av_half_N0) and Nmax > 3:
                    N0 = int(Nmax/2)
        
        # Updates the value of n
        n_ans = n
        n = n + sgn*step
        i += 1 # updates iteration number
        
        # Makes sure we never have f>1
        if n > N:
            step = N - n_ans
            n = N
        
        # Makes sure we never have f=0
        if n <= 0:
            step = n_ans - 1
            n = 1
        
        # If this n was already tested, take the previously calculated values from History
        if n in History['nf']:
            j = History['nf'].index(n)
            
            r = History['rf_hist'][j]
            r_std = History['r_stdf_hist'][j]
            psidot = History['psidotf_hist'][j]
            psidot_std = History['psidot_stdf_hist'][j]
        
        # If this K is new, compute its corresponding synchronization parameters
        else:
            vec = np.zeros(N)
            aux = np.random.randint(0, N-1, n)
            vec[aux] = F*np.ones(n)
            network.F = vec
            sync_par = AverageOrderPar(network, N0, t0, tf, K, dt=dt)
        
            r = sync_par['r']
            r_std = sync_par['r_std']
            psidot = sync_par['psidot']
            psidot_std = sync_par['psidot_std']
        
        # if any condition is not met, increase the number of forced oscillators (sgn=1)
        if r < r_f or np.absolute(psidot) > psidot_f or \
        r_std > error_r and psidot_std > error_psidot:
            
            d = sgn # if the previous step direction is the same, d=1. Else, d=-1
            sgn = 1
        
        # if all conditions are met, try decreasing n (sgn<0) so we may get its smallest value
        else:
            d = -sgn # if the previous move was to increase K, then d=-1. Else, d=1
            sgn = -1
            
        # if we switched the step direction (d=-1), cut the step size in half
        if d < 0:
            step = int(step/2)
        
        # If it's the last step, choose between the last result and the previous one
        if step <= epsilon/2 or i==max_iter:
            
            # if the last step doesn't meet all the criteria but the previous one does, choose the latter
            if sgn > 0 and d < 0:
                n = n_ans
                r = History['rf_hist'][-1]
                psidot = History['psidotf_hist'][-1]
                r_std = History['r_stdf_hist'][-1]
                psidot_std = History['psidot_stdf_hist'][-1]
            
        # Save the parameters obtained in this step at the History dictionary
        History['nf'].append(n)
        History['rf_hist'].append(r)
        History['psidotf_hist'].append(psidot)
        History['r_stdf_hist'].append(r_std)
        History['psidot_stdf_hist'].append(psidot_std)
        History['step_hist'].append(step)
        History['N0'].append(N0)
                 
    fcrit = n/N
    crit_par = {'fcrit': fcrit, 'nf': n, 'rf': r, 'psidotf': psidot, 'r_stdf': r_std, \
                'psidot_stdf': psidot_std, 'iterations': i}
    
    return fcrit, crit_par, History

def ForcedOrderParameter(network, K, t0, tf, F0, Ff=None, dF=.5, N0=1, dt=.5, f=None, Nf=1):
    
    """ Calculates the synchronization parameters for the given value(s) of F
    
    Parameters
    ----------
    network : StuartLandau or KuramotoNetwork object
        Network of N oscillators
    K : float
        Coupling strength (greater than the full synchronization coupling)
    t0 : float
        Beggining of the time interval on which we evaluate the sync parameters
    tf : float
        End of the time interval
    F0 : int, float or ndarray
        Force values for which to calculate the sync parameters (or beginning of force interval)
    Ff : float
        End of force interval. If Ff=None, use only the value(s) given in F0
    dF : float
        Step size of force interval
    N0 : int
        Number of initial states in which to compute the parameters
    dt : float
        Size of the steps inside the time interval
    f : float
        Fraction of (random) oscillators to be forced. If f is None, use current network configuration
    Nf : int
        Number of network configurations with forced fraction f to be tested

    Returns
    -------
    forced_par : dict
        Dictionary with order parameter, mean phase, mean phase velocity, and all its errors for
        each value of the force magnitude F
    """

    N = network.N
    n = np.size(network.forced_osc)

    force_vec = np.zeros(N)
    aux = network.forced_osc
    force_vec[aux] = np.ones(n)

    if f is not None:
        n = int(f*N)

    if Ff==None:
        if type(F0)==int or type(F0)==float:
            F = np.array([F0])
        else:
            F = F0
    else:
        F = np.arange(F0,Ff,dF)
    
    NF = np.size(F)
    
    r = np.zeros((Nf, NF))
    r_std = np.zeros((Nf, NF))
    psi = np.zeros((Nf, NF))
    psi_std = np.zeros((Nf, NF))
    psidot = np.zeros((Nf, NF))
    psidot_std = np.zeros((Nf, NF))

    for i in range(Nf):
        
        if f is not None:
            force_vec = np.zeros(N)
            aux = np.random.randint(0, N-1, n)
            force_vec[aux] = np.ones(n)
            
        for j in range(NF):

            network.F = F[j]*force_vec
              
            # Evolves synchronization parameters
            sync_par = AverageOrderPar(network, N0, t0, tf, K, dt=dt)
        
            r[i][j] = sync_par['r']
            r_std[i][j] = sync_par['r_std'] + r[i][j]
            psi[i][j] = sync_par['psi']
            psi_std[i][j] = sync_par['psi_std'] + psi[i][j]
            psidot[i][j] = sync_par['psidot']
            psidot_std[i][j] = sync_par['psidot_std'] + psidot[i][j]
    
    r = np.average(r, axis=0)
    psi = np.average(psi, axis=0)
    psidot = np.average(psidot, axis=0)

    r_std = np.average(r_std, axis=0) - r
    psi_std = np.average(psi_std, axis=0) - psi
    psidot_std = np.average(psidot_std, axis=0) - psidot
    
    if NF==1:
        F = F[0]
        r = r[0]
        r_std= r_std[0]
        psi = psi[0]
        psi_std = psi_std[0]
        psidot = psidot[0]
        psidot_std = psidot_std[0]
    
    forced_par = {'F': F,
                  'r': r,
                  'r_std': r_std,
                  'psi': psi,
                  'psi_std': psi_std,
                  'psidot': psidot,
                  'psidot_std': psidot_std
                 }
    
    return forced_par

def CriticalForce(network, K, F0, t0, tf, dt=0.5, f=None, step0=1, N0=1, r_f=.95, psidot_f=.01, \
                  error_r=.005, error_psidot=0.005, epsilon=.1, max_iter=30, av_half_N0=3, av_N0=2):
    
    """ Calculates the minimum force magnitude for the system to synchronize with the external force
    
    Parameters
    ----------
    network : StuartLandau or KuramotoNetwork object
        Network of N oscillators
    K : float
        Coupling strength
    F0 : float
        Initial guess for the critical force
    t0 : float
        Beggining of the time interval on which we evaluate the sync parameters
    tf : float
        End of the time interval
    dt : float
        Size of the steps inside the time interval
    f : float
        Fraction of (random) oscillators to be forced. If f is None, use current network configuration
        
    Returns
    -------
    F : float
        Cirtical force
    crit_par : dict
        Dictionary with all calculated parameters for the critical force
    History : dict
        History of all coupling values tried and its respective parameters
    """

    network.rot_frame = True
    
    sgn = np.sign(step0) # step direction (negative for left, poditive for right)
    step = np.absolute(step0) # step magnitude
    F = F0 - sgn*step # so that the first loop calculation is for K0
    
    i = 0    
    Nmax = N0
    N0 = 1
    
    N = network.N
    n = np.size(network.forced_osc)

    force_vec = np.zeros(N)
    aux = network.forced_osc
    force_vec[aux] = np.ones(n)

    if f is not None:
        n = int(f*N)
        force_vec = np.zeros(N)
        aux = np.random.randint(0, N-1, n)
        force_vec[aux] = np.ones(n)
        
    History = {'F_hist': [], 'rf_hist': [], 'psidotf_hist': [], 'r_stdf_hist': [], \
               'psidot_stdf_hist': [], 'step_hist': [], 'N0': []}
    
    while step > epsilon/2 and i < max_iter:
        
        i += 1 # updates iteration number
        F_ans = F
        F = F + sgn*step # Adds the current step to the coupling constant
        
        if F < 0:
            step = step + F
            F = epsilon
        
        if Nmax > 1:
            if step <= epsilon*(2**av_N0):
                N0 = Nmax
            else:
                if step <= epsilon*(2**av_half_N0) and Nmax > 3:
                    N0 = int(Nmax/2)
        
        # If this F was already tested, take the previously calculated values from History
        if F in History['F_hist']:
            j = History['F_hist'].index(F)
            
            r = History['rf_hist'][j]
            r_std = History['r_stdf_hist'][j]
            psidot = History['psidotf_hist'][j]
            psidot_std = History['psidot_stdf_hist'][j]
        
        # If this F is new, calculate the synchronization parameters
        else:           
            network.F = F*force_vec
            sync_par = AverageOrderPar(network, N0, t0, tf, K, dt=dt)
        
            r = sync_par['r']
            r_std = sync_par['r_std']
            psidot = sync_par['psidot']
            psidot_std = sync_par['psidot_std']
        
        # if any condition is not met, increase the coupling (sgn=1)
        if r < r_f or np.absolute(psidot) > psidot_f or \
        r_std > error_r and psidot_std > error_psidot:
            
            d = sgn # if the previous step direction is the same, d=1. Else, d=-1
            sgn = 1
        
        # if all conditions are met, try decreasing K so we may get its smallest value
        else:
            d = -sgn # if the previous move was to increase K, then d=-1. Else, d=1
            sgn = -1
            
        # if we switched the step direction (d=-1), cut the step size in half
        if d < 0:
            step = step/2
        
        # If it's the last step, choose between the last result and the previous one
        if step <= epsilon/2 or i==max_iter:
            
            # if the last step doesn't meet all the criteria but the previous one does, choose the latter
            if sgn > 0 and d < 0:
                F = F_ans
                r = History['rf_hist'][-1]
                psidot = History['psidotf_hist'][-1]
                r_std = History['r_stdf_hist'][-1]
                psidot_std = History['psidot_stdf_hist'][-1]
        
        # Save the parameters obtained in this step at the History dictionary
        History['F_hist'].append(F)
        History['rf_hist'].append(r)
        History['psidotf_hist'].append(psidot)
        History['r_stdf_hist'].append(r_std)
        History['psidot_stdf_hist'].append(psidot_std)
        History['step_hist'].append(step)
        History['N0'].append(N0)
        
    force_par = {'F': F, 'rf': r, 'psidotf': psidot, 'r_stdf': r_std, 'psidot_stdf': psidot_std, 'iterations': i}
    
    return F, force_par, History

def RandomCoupling(network, Karray, direc=False, norm=True, dist_type=None, shape=None, scale=None, loc=None):
    
    """ Calculates the order parameter (as well as other synchronization parameters) of the 
    given network for one or seveal coupling constants averaged in the given time interval 
    
    Parameters
    ----------
    network : StuartLandau or KuramotoNetwork object
        Initial network
    Karray : ndarray of shape (n)
        Distribution vector
    direct : bool
        False if the output network is undirected, True otherwise
    normalize : bool
        True if the desired mean value is one (normalized)
        
    Returns
    -------
    network : StuartLandau or KuramotoNetwork object
        Initial network with gamma distribuited edge weights
    Kmean : float
        Average value of the gamma distribuition
    Kstd : float
        Standard deviation of the gamma distribution
    """
    
    Ne = network.Ne
    nz = network.edges
    
    # Takes away the edges with i>j for the undirected case
    if direc==False:
        aux = []
        for k in range(Ne):
            [i, j] = nz[k]
            if i<j:
                aux.append([i,j])
        nz = np.array(aux)
        Ne = np.size(nz,0)
    
    # Makes sure Karray is the right size
    Karray = Karray[0:Ne]
    
    # Calculates distribution parameters
    Kmean = np.mean(Karray)
    Kstd = np.std(Karray)
    
    if norm:
        Karray = Karray/Kmean
        Kmean = 1
        Kstd = Kstd/Kmean
    
    for k in range(Ne):
        [i, j] = nz[k]
        network.A[i,j] = Karray[k]
        
        # Assures A is symmetric for the undirected case
        if direc==False:
            network.A[j,i] = Karray[k]
    
    network.K = 1
    network.dist = {'type' : dist_type, 
                'shape' : shape,
                'scale' : scale,
                'loc' : loc,
                'mean' : Kmean,
                'std' : Kstd}
    
    return network