import numpy as np

from kinematics.transformations import skew_symmetric
from scipy.linalg import expm

def point_rocket(x, u, w=[1,1,1]):
    """
    w: constant angular velocity of the planet (w1, w2, w3)
    g: uniform gravity vector (g1, g2, g3)
    
    """
    A = np.zeros((6,6))
    A[:3,3:] = np.eye(3)
    A[3:,:3] = - skew_symmetric(w) @ skew_symmetric(w)
    A[3:,3:] = - 2 * skew_symmetric(w)

    B = np.zeros((6,3))
    B[3:,:] = np.eye(3)

    xdot = A @ x + B @ u


    return xdot



def point_rocket_dynamics(w, alpha=5e-4):
    """
    w: constant angular velocity of the planet (w1, w2, w3)
    g: uniform gravity vector (g1, g2, g3)
    
    """
    A = np.zeros((6,6))
    A[:3,3:] = np.eye(3)
    A[3:6,:3] = - skew_symmetric(w) @ skew_symmetric(w)
    A[3:6,3:] = - 2 * skew_symmetric(w)

    B = np.zeros((6,3))
    B[3,:] = 1
    B[4,:] = 1
    B[5,:] = 1


    return A, B

   
def get_discrete_dynamics(Ac, Bc, dt):
   
    an = Ac.shape[1]
    bn = Bc.shape[1]
   
    dd = np.zeros((an+bn,an+bn))
    dd[0:an,0:an] = Ac
    dd[0:an,an:an+bn] = Bc
    exp_mx = expm(dd*dt)

    Ad = exp_mx[0:an,0:an]
    Bd = exp_mx[0:an,an:an+bn]

    return Ad, Bd