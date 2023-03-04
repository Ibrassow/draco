import numpy as np
import matplotlib.pyplot as plt

from astro_constants import earth
from scipy.linalg import expm


def central_two_body(state, u):
    """ 
    Version of the two-body problem treating one body as an immobile source of a force acting on the other.
    This assumes the central object is much more massive than the moving one.

    'state' is a 6x1 vector of x,y,z positions and velocities of the orbiting body. 
    """

    r = state[:3]
    a = - earth['mu'] * r / np.linalg.norm(r)**3

    return np.array([ state[3], state[4], state[5], a[0], a[1], a[2] ])



def clohessy_wilthsire(x, u, t, mu=3.986004418 * 10**(14), a=42164000):


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
    A = np.array([[4-3*np.cos(n*t),0,0,(1/n) * np.sin(n*t), (2/n)*(1-np.cos(n*t)),0], 
                [6*(np.sin(n*t)-n*t),1,0,-(2/n)*(1-np.cos(n*t)),(1/n)*(4*np.sin(n*t)-3*n*t),0],
                [0,0,np.cos(n*t),0,0,(1/n)*np.sin(n*t)],
                [3*n*np.sin(n*t),0,0,np.cos(n*t),2*np.sin(n*t),0],
                [-6*n*(1-np.cos(n*t)),0,0,-2*np.sin(n*t),4*np.cos(n*t)-3,0],
                [0,0,-n*np.sin(n*t),0,0,np.cos(n*t)]])

    x_dot = np.dot(A,x)

    return x_dot



def clohessy_wilthsire_new(x, u, t, mu=3.986004418 * 10**(14), a=42164000, delta_t=0.1):


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
                [0,0,-n*2,0,0,0]])

    # out: ## input : [xdot, ydot, zdot, xdotdot, ydotdot, zdotdot]
    x_dot = np.dot(A,x)
    #q = np.array([[A, 0],[0,0]])
    exp_d = expm(A*delta_t)
    pos = np.dot(exp_d,x)

    return x_dot, pos







if __name__ == '__main__':
    from integrator import rk4
    plt.style.use('dark_background')
    dt = 60.0           # seconds
    tspan = 24 * 60.0 * 60.0 # seconds
    steps = int(tspan/dt)

    r0 = earth["R_GEO"]
    v0 = (earth["mu"]/r0)**0.5
    initial_state = [r0,0,0,0,v0,0]
    states = np.zeros((steps, 6))
    timesteps = np.zeros((steps,1))
    states[0] = initial_state


    chaser_states = np.zeros((steps, 6))
    chaser_states[0] = initial_state
    chaser_states[0,0] += 0
    chaser_states[0,1] += -50
    chaser_states[0,2] += 0
    chaser_states[0,3] += 10
    chaser_states[0,4] += 0
    chaser_states[0,5] += 100



    for step in range(steps-1):
        states[step+1] = rk4(central_two_body, states[step], 0, dt)
        chaser_states[step+1] = rk4(central_two_body, chaser_states[step], 0, dt)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z') 


    ax.scatter(0,0,0,c='g',label="Central body", marker='o', s=300)

    ax.plot3D(states[:,0], states[:,1], states[:,2], c='b', label="Target - Free fall")
    ax.plot3D(chaser_states[:,0], chaser_states[:,1], chaser_states[:,2], c='y', label="Chaser - Free fall")

    plt.legend()
    plt.show()

