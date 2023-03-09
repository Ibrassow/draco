import numpy as np
from transformations import mat_exp_se3, adjoint


def space_forward_kinematics(M_home, twist):
    """
    Computes the forward kinematics in space frame
    """

    T = np.array(M_home)
    for i in range(len(twist)-1, -1, -1):
        T = mat_exp_se3(twist[i]) @ T
    return T



def body_forward_kinematics(M_home, twist):
    """
    Computes the forward kinematics in body frame
    """

    T = np.array(M_home)
    for i in range(len(twist)):
        T =  T @ mat_exp_se3(twist[i]) 
    return T


def space_jacobian(screw_list, joint_values):
    """
    Computes the space Jacobian

    screw_list -> axis as columns (.T), must be of size 6 x n
    """

    Js = np.array(screw_list).copy()

    T = np.eye(4)

    for i in range(1, len(joint_values)):
        twist = screw_list[:,i]*joint_values[i]
        print(twist)
        T = T @ mat_exp_se3(twist) 
        Js[:, i] = adjoint(T) @ screw_list[:,i]

    return Js


import modern_robotics as mr

# Screw axis of the joints
S1 = np.array([0,0,1,0,0,0])
S2 = np.array([1,0,0,0,2,0])
S3 = np.array([1,0,0,0,2,-2])

# Joint value µ
# TODO something is wrong here
q1 = 75*np.pi/180
q2 = 32*np.pi/180
q3 = 28*np.pi/180

Q = np.array([q1,q2,q3])

print("Twists")
V1=S1*q1
V2=S2*q2
V3=S3*q3

S = np.array([S1,S2,S3])
V = np.array([V1,V2,V3])

ii = space_jacobian(S.T, Q)
mm = mr.JacobianSpace(S.T, Q)

print(ii)
print(mm)