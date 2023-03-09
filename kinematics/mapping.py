
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



def exp_map_so3(w, theta):
    """
    Returns the exponential map from so(3) to SO(3)
    """
    return exp_map_so3(w*theta)




def exp_map_so3(wtheta):
    """
    Returns the exponential map from so(3) to SO(3)
    """
    theta = np.linalg.norm(wtheta)
    if theta > 0:
        w_hat = skew_symmetric(wtheta/theta)
        R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat
    else:
        R = np.eye(3)
    return R


def check_rot(mtx):
    if mtx.shape != (3,3):
        return False
    elif np.linalg.det(mtx) != 1:
        return False
    elif ((np.transpose(mtx) @ mtx) != np.eye(3)).any():
        return False
    else:
        return True
    


def exp_map_se3_from_units(theta, w, v):
    """
    Returns the exponential map from se(3) to SE(3)
    """
    return exp_map_se3_from_vels(theta*w, theta*v)



def exp_map_se3(twist):
    """
    Returns the exponential map from se(3) to SE(3)
    """
    return exp_map_se3_from_vels(twist[:3], twist[3:])


def exp_map_se3_from_vels(theta_w, theta_v):
    """
    Returns the exponential map from se(3) to SE(3)
    """

    theta = np.linalg.norm(theta_w)

    if (theta>0):
        #w_hat = skew_symmetric(theta_w)
        nwhat = skew_symmetric(theta_w/theta)
        R = exp_map_so3(theta_w)
        p = (np.eye(3) * theta + (1 - np.cos(theta))*nwhat + (theta - np.sin(theta)) * (nwhat @ nwhat)) @ (theta_v/theta)
    else:
        R = np.eye(3)
        p = theta_v


    res = np.zeros((4,4))
    res[3,3] = 1  
    # return R, p
    res[0:3,0:3] = R
    res[3, 0:3] = p
    
    return res





if __name__ == '__main__':


    tw = np.array([4,8,1])
    tv = np.array([4,7,1])

    ii = exp_map_se3_from_vels(tw, tv)
