"""
@author: ibrahima
"""

import numpy as np

class EKF:

    """
    Extended Kalman Filter

    x_hat    Estimated state (nx x 1)
    P        Estimate covariance matrix (nx x nx)
    u        Control Input vector (nu x 1)
    z        Measurement (nz x 1)
    f        Propagation function 
    h        Observation function
    F        Jacobian of the propagation function 
    H        Jacobian of the observation function
    Q        Process noise covariance matrix (nx x nx)
    R        Measurement noise covariance matrix (nz x nz)
    """


    def __init__(self, x0, propagation, observation, jac_propagation, jac_observation, process_noise, measurement_noise):
        self.f = propagation
        self.h = observation
        self.F = jac_propagation
        self.H = jac_observation
        self.Q = process_noise
        self.R = measurement_noise
        self.x_hat = x0
        self.z_hat = None


    def predict(self):
        ## Propagate the estimate 
        self.x_hat = self.f(self.x_hat)
        ## Propagate the covariance 
        F = self.jac_propagation(self.x_hat)
        self.P = F @ self.P @ F.T + self.Q 
        ## Predict the measurement using the predicted state estimate.
        self.z_hat = self.h(self.x_hat)

        

    def update(self, z):
        ## Innovation vector
        y = z - self.z_hat
        ## State-innovation and innovation covariances 
        H = self.obs_propagation(self.x_hat)
        Pxy = self.P @ H.T
        Pyy = H @ self.P @ self.P.T + self.R
        ## Kalman gain
        K = Pxy @ self.linalg.inv(Pyy)
        ## Correct the estimate 
        self.x_hat = self.x_hat + K @ y
        ## Correct covariance 
        self.P = self.P - K @ H @ self.P






if __name__ == "__name__":

    pass