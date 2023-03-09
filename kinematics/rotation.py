import numpy as np
from numpy import sin, cos






def rot(axis, angle):

    if axis == 'x':
        rot = np.array([[1,0,0], [0,cos(angle),-sin(angle)], [0,sin(angle),cos(angle)]])
    elif axis == 'y':
        rot = np.array([[cos(angle), 0, sin(angle)], [0,1,0], [-sin(angle), 0, cos(angle)]])
    elif axis == 'z':
        rot = np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0,0,1]])                       
    return rot

def check_rot(mtx):

    if mtx.shape != (3,3):
        return False
    elif np.linalg.det(mtx) != 1:
        return False
    elif ((np.transpose(mtx) @ mtx) != np.eye(3)).any():
        return False
    else:
        return True



def zyx_euler(alpha, beta, gamma):
    return rot('z', alpha) * rot('y', beta) * rot('x', gamma)


def DH(alpha, a, d, theta):

    Rot_z_theta = np.array([[cos(theta),-sin(theta), 0, 0], [sin(theta), cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                
    Tran_z_d = [[1,0,0,0],[0,1,0,0],[0,0,1,d],[0,0,0,1]]
    
    Tran_x_a = [[1,0,0,a],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
              
    Rot_x_alpha = [[1,0,0,0], [0, cos(alpha), -sin(alpha), 0], [0,sin(alpha),cos(alpha),0], [0,0,0,1]]
           
    return Rot_z_theta @ Tran_z_d @ Tran_x_a @ Rot_x_alpha



if __name__ == '__main__':

    rr = np.array([[1,0,0],[0,1,2],[0,7,1]])

    print(check_rot(rr))
    print(np.linalg.det(rr))

