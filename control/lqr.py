"""

TODO
- LQR: Riccati 
- Test w/ CW equations
- Test with 2 link manipulator in joint space 
- Test with hovering 


"""

import numpy as np
import scipy.linalg



"""def lqr(A, B, Q, R, N):


    return K"""


def dlqr(A, B, Q, R, N=0):
    """
    Computes the optimal gain matrix K such that the state-feedback law u[k] = - K x[k] minimizes
    the cost function J(u) = SUM(k=1:N)[ x[k]' Q x[k] + u[k]' R u[k] + 2 x[k] N x[k] ]
    """
    if N == 0:
        N = np.zeros((A.shape[0],B.shape[1]))
        
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    K = np.matrix(scipy.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A + N.T))
    return K




