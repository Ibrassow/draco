import numpy as np


def skew_symmetric(w):
    """
    Returns the skew symmetric form of a numpy array.
    w --> [w]
    """
    #TODO check
    
    return np.array([[0, -w[2], w[1]], 
                     [w[2], 0, -w[0]], 
                     [-w[1], w[0], 0]])


def mat_exp_so3(w, theta):
    """
    Returns the exponential map from so(3) to SO(3)
    """
    return mat_exp_so3(w*theta)


def mat_exp_so3(wtheta):
    """
    Returns the exponential map from so(3) to SO(3)
    """
    theta = np.linalg.norm(wtheta)
    if theta > 1e-6:
        w_hat = skew_symmetric(wtheta/theta)
        R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat
    else:
        R = np.eye(3)
    return R





def check_rotation_matrix(mtx):
    if mtx.shape != (3,3):
        return False
    elif np.linalg.det(mtx) != 1:
        return False
    elif ((np.transpose(mtx) @ mtx) != np.eye(3)).any():
        return False
    else:
        return True
    





def mat_exp_se3_from_units(theta, w, v):
    """
    Returns the matrix exponential of an se(3) representation (se(3) to SE(3))
    """
    V = np.zeros(6)
    V[:3] = theta*w
    v[3:] = theta*v
    return mat_exp_se3(V)




def mat_exp_se3(twist):
    """
    Returns the matrix exponential of an se(3) representation (se(3) to SE(3))
    """
    mat_se3 = twist_to_se3(twist)
    omg_theta = twist[0:3]
    theta = np.linalg.norm(omg_theta)

    if (theta>1e-6):
        omg_mat = mat_se3[0:3,0:3] / theta
        R = mat_exp_so3(twist[0:3])
        p = np.dot(np.eye(3) * theta + (1 - np.cos(theta)) * omg_mat + (theta - np.sin(theta)) * np.dot(omg_mat,omg_mat),mat_se3[0: 3, 3] / theta) 
    else:
        R = np.eye(3)
        p = mat_se3[0:3, 3]

    SE3 = np.zeros((4,4))
    SE3[3,3] = 1
  
    # return R, p
    SE3[0:3,0:3] = R
    SE3[0:3, 3] = p
    
    return SE3



def twist_to_se3(twist):
    """
    Converts the spatial velocity vector into a 4x4 matrix in se(3)
    """
    mat_se3 = np.zeros((4,4))
    mat_se3[0:3,0:3] = skew_symmetric(twist[:3])
    mat_se3[0:3, 3] = twist[3:6]
    return mat_se3

def se3_to_twist(mat_se3):
    """
    Converts back a 4x4 matrix in se(3) into the spatial velocity vector
    """
    return np.array([mat_se3[2,1],mat_se3[0,2],mat_se3[1,0],mat_se3[0,3],mat_se3[1,3],mat_se3[2,3]])



def HT_to_RP(T):
    return T[0: 3, 0: 3], T[0: 3, 3]

def RP_to_HT(R, p):
    T = np.zeros((4,4))
    T[0: 3, 0: 3] = R
    T[0: 3, 3] = p
    T[3,3] = 1
    return T



if __name__ == '__main__':

    #dd
    u = None
