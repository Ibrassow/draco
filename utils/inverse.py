import numpy as np



def dls_pinv(J, damping=1e-6):
    J_pinv = np.linalg.pinv(J.T.dot(J) + damping**2 * np.eye(J.shape[1])).dot(J.T)
    return J_pinv


def SR_inverse_nakamura(J, k=0.002):
    # SIngularity robust inverse jacobian - https://asmedigitalcollection.asme.org/dynamicsystems/article/108/3/163/425826/Inverse-Kinematic-Solutions-With-Singularity
    return J.T @ np.linalg.inv(J @ J.T + k * np.eye(J.shape[0]))