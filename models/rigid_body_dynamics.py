import numpy as np
from kinematics.transformations import *

def rigid_body_dynamics(params, x, u):
    """
    Decouple translational and rotational dynamics
    """
    r = x[0:2]  # Position
    q = x[3:6]  # Attitude (quat)
    r_dot = x[7:8]  # Linear Velocity
    omega = x[10:12]  # Angular Velocity

    F = u[0:2]
    tau = u[3:5]

    H = np.vstack((np.zeros((1, 3)), np.eye(3)))

    q_dot = 0.5 * L(q) @ H.dot(omega)

    r_ddot = (1 / params['m']) * F
    omega_dot = np.linalg.inv(params['J']).dot(tau - skew_symmetric(omega).dot(params['J']).dot(omega))

    x_dot = np.concatenate((r_dot, q_dot, r_ddot, omega_dot))
    return x_dot


def rigid_body_dynamics_mrp(params, x, u):
    """
    Decouple translational and rotational dynamics
    """
    r = x[0:2]  # Position
    mrp = x[3:5]  # Attitude (MRP)
    r_dot = x[6:8]  # Linear Velocity
    omega = x[9:11]  # Angular Velocity

    F = u[0:2]
    tau = u[3:5]

    q = quat_from_mrp(mrp)
    H = np.vstack((np.zeros((1, 3)), np.eye(3)))

    qq = 0.5 * L(q) @ H.dot(omega)
    q_dot = mrp_from_quat(qq)

    r_ddot = (1 / params['m']) * F
    omega_dot = np.linalg.inv(params['J']).dot(tau - skew_symmetric(omega).dot(params['J']).dot(omega))

    x_dot = np.concatenate((r_dot, q_dot, r_ddot, omega_dot))
    return x_dot



def keplerian_central_body_dynamics(params, x, u):
    """
    Decouple translational and rotational dynamics
    """
    r = x[0:2]  # Position
    r_dot = x[3:5]  # Linear Velocity
    F = u

    r_ddot = (1 / params['m']) * F - (params['mu'] / np.linalg.norm(r) ** 3) * r
    x_dot = np.concatenate((r_dot, r_ddot))
    return x_dot


def mrp_kinematics(sigma, omega):
    """
    Analytical Mechanics of Space System eq. 3.154 p122, Hanspeter Schaub, John L. Junkins
    """
    sigma1, sigma2, sigma3 = sigma

    M = np.array([[1 - np.dot(sigma, sigma) + 2 * sigma1 ** 2, 2 * (sigma1 * sigma2 - sigma3), 2 * (sigma1 * sigma3 + sigma2)],
                  [2 * (sigma2 * sigma1 + sigma3), 1 - np.dot(sigma, sigma) + 2 * sigma2 ** 2, 2 * (sigma2 * sigma3 + sigma1)],
                  [2 * (sigma3 * sigma1 - sigma2), 2 * (sigma3 * sigma2 + sigma1), 1 - np.dot(sigma, sigma) + 2 * sigma3 ** 2]])

    sigma_dot = 0.25 * M.dot(omega)

    return sigma_dot



def attitude_mrp_rbd(params, x, tau):
    """
    Rotational dynamics only
    """
    mrp = x[0:2]  # Attitude (MRP)
    omega = x[3:5]  # Angular Velocity

    mrp_dot = mrp_kinematics(mrp, omega)
    omega_dot = np.linalg.inv(params['J']).dot(tau - skew_symmetric(omega).dot(params['J']).dot(omega))

    return np.concatenate((mrp_dot, omega_dot))



def nonlinear_relative_keplerian_dynamics(params, X, F):
    x = X[0]
    y = X[1]
    z = X[2]
    x_dot = X[3]
    y_dot = X[4]
    z_dot = X[5]

    rt = params['target']['sma']
    rtdot = np.sqrt(params['earth']['mu'] / rt)

    rc = np.sqrt((rt + x) ** 2 + y ** 2 + z ** 2)
    theta_dot = np.sqrt(params['earth']['mu'] / rt ** 3)

    x_ddot = - (params['mu'] / rc ** 3) * (rt + x) + theta_dot ** 2 * x + 2 * theta_dot * (y_dot - y * (rtdot / rt)) + params['mu'] / rt ** 2 + F[0] / params['m']
    y_ddot = - (params['mu'] / rc ** 3) * y + theta_dot ** 2 * y - 2 * theta_dot * (x_dot - x * (rtdot / rt)) + F[1] / params['m']
    z_ddot = - (params['mu'] / rc ** 3) * z + F[2] / params['m']

    return np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])
