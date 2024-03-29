import numpy as np


def skew_symmetric(w):
    """
    Returns the skew symmetric form of a numpy array.
    w --> [w]
    """
    
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
    mat_se3[:3, 3] = twist[3:6]
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
    T[:3, :3] = R
    T[:3, 3] = p
    T[3,3] = 1
    return T



def adjoint(ht):
    """
    Computes the adjoint representation (6x6) of an homogeneous transformation matrix (SE(3))
    """
    R, p = HT_to_RP(ht)
    adj = np.zeros((6,6))
    adj[:3,:3] = R
    adj[3:,3:] = R
    adj[3:,:3] = skew_symmetric(p) @ R

    return adj

### Quaternions 
# https://roboticexplorationlab.org/papers/planning_with_attitude.pdf

def L(q):
    """
    Left-multiply
    """
    L = np.zeros((4,4))
    L[0,0] = q[0]
    L[0,1:] = -q[1:].T
    L[1:,0] = q[1:]
    L[1:,1:] = q[0]*np.eye(3) + skew_symmetric(q[1:])
    return L

def R(q):
    """
    Right-multiply
    """
    R = np.zeros((4,4))
    R[0,0] = q[0]
    R[0,1:] = -q[1:].T
    R[1:,0] = q[1:]
    R[1:,1:] = q[0]*np.eye(3) - skew_symmetric(q[1:])
    return R

def conj(q):
    """
    Inverse of a unit quaternion is its conjugate, i.e. same quaternion with a negated vector part 
    """
    qr = np.zeros(4)
    qr[0] = q[0]
    qr[1:] = - q[1:]
    return qr

def rotm2quat(r):
    q = np.zeros(4)
    q[0] = 0.5 * np.sqrt(1 + r[0,0] + r[1,1] + r[2,2])
    q[1] = (1/(4*q[0])) * (r[2][1] - r[1][2])
    q[2] = (1/(4*q[0])) * (r[0][2] - r[2][0])
    q[3] = (1/(4*q[0])) * (r[1][0] - r[0][1])
    return np.array(q)




def angular_vel_from_quat(q1, q2, dt):
    """
    https://mariogc.com/post/angular-velocity-quaternions/
    """
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])



def angvel_from_quat(q1, q2, dt):
    # TODO - must ensure they are diff
    delta_q = L(conj(q1)) @ q2
    delta_q = delta_q / np.linalg.norm(delta_q)

    angle = 2 * np.arccos(delta_q[0])
    axis = delta_q[1:] / np.sin(0.5 * angle)
    return axis * angle / dt





def mrp_from_quat(q):
    """
    Modified Rodriguez parameter inverse mapping
    """
    return q[1:3] / (1 + q[0])

def quat_from_mrp(mrp):
    q = np.concatenate(([1 - np.dot(mrp,mrp)], 2*mrp))
    return (1/(1+np.dot(mrp,mrp))) * q


def dcm_from_q(q):
    norm = np.linalg.norm(q)
    q0, q1, q2, q3 = q / norm if norm != 0 else q

    # DCM
    Q = np.array([
        [2*q1**2 + 2*q0**2 - 1,   2*(q1*q2 - q3*q0),   2*(q1*q3 + q2*q0)],
        [2*(q1*q2 + q3*q0),       2*q2**2 + 2*q0**2 - 1,   2*(q2*q3 - q1*q0)],
        [2*(q1*q3 - q2*q0),       2*(q2*q3 + q1*q0),   2*q3**2 + 2*q0**2 - 1]
    ])

    return Q


def quat2axisangle(q):
    """
    quat wxyz
    """
    axis = np.zeros(3)
    angle = 2 * np.arccos(q[0])
    axis = q[1:]/np.sqrt(1 - q[0]*q[0])
    return axis*angle





