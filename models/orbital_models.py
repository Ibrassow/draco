import numpy as np
import matplotlib.pyplot as plt


from scipy.linalg import expm



earth = {
    "mu": 3.986004418 * 10**(14), # standard gravitational parameter ==> mu = GM
    "G":  6.67430*10**(-11), # N * m2 * kg-2
    "M": 5.9722*10**(24), #kg
    "R": 6378*10**3, #m
    "R_GEO": 42164000, #m
    "R_LEO_ub": 8413*10**3 #m
}




"""def central_two_body(state, u):
    ""
    Version of the two-body problem treating one body as an immobile source of a force acting on the other.
    This assumes the central object is much more massive than the moving one.

    'state' is a 6x1 vector of x,y,z positions and velocities of the orbiting body. 
    ""

    r = state[:3]
    a = - earth['mu'] * r / np.linalg.norm(r)**3

    return np.array([ state[3], state[4], state[5], a[0], a[1], a[2] ])"""



def clohessy_wilthsire(x, u, t, mu=3.986004418 * 10**(14), a=42164000):


    """
    Discrete model
        x-axis is along the radius vector of the target spacecraft, the z-axis
        is along the angular momentum vector of the target spacecraft, and the y-axis
        completes the right handed system. With this de...nition, the central body
        is towards the negative x direction and the y-axis points along the velocity
        vector of the target spacecraft. Motion along y-axis is considered ‘along-track’,
        and motion along the positive and negative z-axis is considered ‘out-of-plane’
        motion.
    """
    #mu = 3.986004418 * 10**(14) # standard gravitational parameter ==> mu = GM
    #a = 42164000 # radius of the target body's circular orbit (here radius GEO, which is 42,164 km)
    
    n = np.sqrt(mu/a**3)
    A = np.array([[4-3*np.cos(n*t),0,0,(1/n) * np.sin(n*t), (2/n)*(1-np.cos(n*t)),0], 
                [6*(np.sin(n*t)-n*t),1,0,-(2/n)*(1-np.cos(n*t)),(1/n)*(4*np.sin(n*t)-3*n*t),0],
                [0,0,np.cos(n*t),0,0,(1/n)*np.sin(n*t)],
                [3*n*np.sin(n*t),0,0,np.cos(n*t),2*np.sin(n*t),0],
                [-6*n*(1-np.cos(n*t)),0,0,-2*np.sin(n*t),4*np.cos(n*t)-3,0],
                [0,0,-n*np.sin(n*t),0,0,np.cos(n*t)]])
    

    x_dot = np.dot(A,x)

    return x_dot 



def d_clohessy_wilthsire(x, u, dt=0.1, mc=1, mu=3.986004418 * 10**(14), a=42164000):


    """
        x-axis is along the radius vector of the target spacecraft, the z-axis
        is along the angular momentum vector of the target spacecraft, and the y-axis
        completes the right handed system. With this de...nition, the central body
        is towards the negative x direction and the y-axis points along the velocity
        vector of the target spacecraft. Motion along y-axis is considered ‘along-track’,
        and motion along the positive and negative z-axis is considered ‘out-of-plane’
        motion.
    """
    #mu = 3.986004418 * 10**(14) # standard gravitational parameter ==> mu = GM
    #a = 42164000 # radius of the target body's circular orbit (here radius GEO, which is 42,164 km)
    
    n = np.sqrt(mu/a**3)
    ## input : [x,y,z, xdot, ydot, zdot]
    A = np.array([[0,0,0,1,0,0], 
                [0,0,0,0,1,0],
                [0,0,0,0,0,1],
                [3*n**2,0,0,0,2*n,0],
                [0,0,0,-2*n,0,0],
                [0,0,-n**2,0,0,0]])
    

    B = (1/mc)*np.array([[0,0,0],
                  [0,0,0],
                  [0,0,0],
                  [1,0,0],
                  [0,1,0],
                  [0,0,1]])


    dd = np.zeros((9,9))
    dd[0:6,0:6] = A
    dd[0:6,6:9] = B
    exp_mx = expm(dd*dt)

    Ad = exp_mx[0:6,0:6]
    Bd = exp_mx[0:6,6:9]


    return np.dot(Ad,x) + np.dot(Bd, u)



def get_cw_discrete_dynamics(dt, mc=1, mu=3.986004418 * 10**(14), a=42164000):

    n = np.sqrt(mu/a**3)

    A = np.array([[0,0,0,1,0,0], 
                [0,0,0,0,1,0],
                [0,0,0,0,0,1],
                [3*n**2,0,0,0,2*n,0],
                [0,0,0,-2*n,0,0],
                [0,0,-n**2,0,0,0]])
    

    B = (1/mc)*np.array([[0,0,0],
                  [0,0,0],
                  [0,0,0],
                  [1,0,0],
                  [0,1,0],
                  [0,0,1]])
    
    dd = np.zeros((9,9))
    dd[0:6,0:6] = A
    dd[0:6,6:9] = B
    exp_mx = expm(dd*dt)

    Ad = exp_mx[0:6,0:6]
    Bd = exp_mx[0:6,6:9]


    return Ad, Bd


