import numpy as np
import quaternion

from transformations import *




def cubic_time_scaling(Tf, t):
    """
    a0 = a1 = a2 = 0, a3 = 10/T**3, a4 = 15/T**4, a5 = 6/T**5
    """
    s = 10 * (t/Tf)**3 - 15 * (t/Tf)**4 + 6 * (t/Tf)**5
    sdot = (30/Tf**3)*(t**2) - (60/Tf**4)*(t**3) + (30/Tf**5)*(t**4)
    sdotdot = (60/Tf**3) * t - (180/Tf**4)*(t**2) + (120/Tf**5)*(t**3)
    return s,sdot,sdotdot


def straight_line_interpolation_joint_space(qi, qf, Tf, N_points):
    diff = Tf/(N_points-1)
    waypoints = np.zeros((N_points,6*3)) ## for each pos, you have q, dq, dqq

    for i in range(N_points):
        s, sdot, sdotdot = cubic_time_scaling(Tf, i*diff)
        waypoints[i][0:6] = s * qf + (1 - s) * qi
        waypoints[i][6:12] = sdot * (qf-qi)
        waypoints[i][12:18] = sdotdot * (qf-qi)
    return waypoints


def generate_straight_joint_space_random_traj(qi, Tf, N_points):
    qf = -np.random.rand(6)*np.pi
    return straight_line_interpolation_joint_space(qi, qf, Tf, N_points)




### Straight line paths 
def generate_straight_line_cartesian_waypoints(current_pose, desired_pose, Tf, N_points, quat=False):
    """
    Point-to-point trajectory generator in cartesian space.
    Generates a straight line trajectory between the current pose and the desired pose as a list of waypoints
    TODO make sure that the trajectory is part of the manipulator's workspace.
    
    """

    if quat==False:
        pstart = current_pose[0:3, 3]
        qstart = rotm2quat(current_pose[:3,:3])
        pend = desired_pose[0:3, 3]
        qend = rotm2quat(desired_pose[:3,:3])
    else:
        pstart = current_pose[:3]
        qstart = current_pose[-4:]
        pend = desired_pose[:3]
        qend = desired_pose[-4:]

    diff = Tf/(N_points-1)

    waypoints = np.zeros((N_points,13))
    """waypoints = {
    "pos": np.zeros((N_points, 3)),
    "vel": np.zeros((N_points, 3)),
    "acc": np.zeros((N_points, 3)),
    "quat": np.zeros((N_points, 4))
    }"""


    for i in range(N_points):
        s, sdot, sdotdot = cubic_time_scaling(Tf, i*diff)
        waypoints[i][0:3] = s * pend + (1 - s) * pstart
        waypoints[i][3:6] = sdot * (pend-pstart)
        waypoints[i][6:9] = sdotdot * (pend-pstart)
        ## quaternions, be careful
        ## Using SLERP here - https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp
        #waypoints[i][3:] = (L(qend) @ conj(qstart))**(s) @ qstart
        qq = quaternion.slerp_evaluate(quaternion.as_quat_array(qstart), quaternion.as_quat_array(qend), s)
        waypoints[i][9] = qq.w
        waypoints[i][10] = qq.x
        waypoints[i][11] = qq.y
        waypoints[i][12] = qq.z

    return waypoints






if __name__ == '__main__':


    N = 10
    qi = np.array([0,0,0,0,0,0])
    qf = np.deg2rad(np.array([45,-90,30,-90,0,0]))
    Tf = 3

    B = straight_line_interpolation_joint_space(qi, qf, Tf, N)

    print(B.shape)


    curr_pos = np.array([1,1,1])
    curr_quat = np.array([1,0,0,0])

    target_pose = np.array([3,2,3, 0.7071068, 0, 0.7071068, 0])

    wp_list = generate_straight_line_cartesian_waypoints(np.concatenate([curr_pos, curr_quat]), target_pose, Tf, N, quat=True)
    print(wp_list)