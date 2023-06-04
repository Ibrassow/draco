import numpy as np
import quaternion

from .transformations import *




def cubic_time_scaling(Tf, t):
    """
    a0 = a1 = a2 = 0, a3 = 10/T**3, a4 = 15/T**4, a5 = 6/T**5
    """
    s = 10 * (t/Tf)**3 - 15 * (t/Tf)**4 + 6 * (t/Tf)**5
    sdot = (30/Tf**3)*(t**2) - (60/Tf**4)*(t**3) + (30/Tf**5)*(t**4)
    sdotdot = (60/Tf**3) * t - (180/Tf**4)*(t**2) + (120/Tf**5)*(t**3)
    return s,sdot,sdotdot



def straight_line_interpolation_joint_space(qi, qf, Tf, N_points, dim=6):
    diff = Tf/(N_points-1)
    waypoints = np.zeros((N_points,dim*3)) ## for each pos, you have q, dq, dqq

    for i in range(N_points):
        s, sdot, sdotdot = cubic_time_scaling(Tf, i*diff)
        waypoints[i][0:1*dim] = s * qf + (1 - s) * qi
        waypoints[i][dim:2*dim] = sdot * (qf-qi)
        waypoints[i][2*dim:3*dim] = sdotdot * (qf-qi)
    return waypoints


def generate_straight_joint_space_random_traj(qi, Tf, N_points, dim=6):
    qf = -np.random.rand(6)*np.pi
    return straight_line_interpolation_joint_space(qi, qf, Tf, N_points, dim=dim)


### Straight line paths 
def straight_line_pose_waypoints(current_pose, desired_pose, Tf, N_points, quat=True):
    """
    Point-to-point trajectory generator in cartesian space.
    Generates a straight line trajectory between the current pose and the desired pose as a list of waypoints
    current_pose, desired_pose: np.array([x, y, z, quaternion_4d(w,x,y,z)]) 
    TODO make sure that the trajectory is part of the manipulator's workspace.
    """
    if quat:
        pstart = current_pose[:3]
        qstart = current_pose[-4:]
        pend = desired_pose[:3]
        qend = desired_pose[-4:]
    else:
        pstart = current_pose[0:3, 3]
        qstart = rotm2quat(current_pose[:3,:3])
        pend = desired_pose[0:3, 3]
        qend = rotm2quat(desired_pose[:3,:3])


    diff = Tf/(N_points-1)

    waypoints = {
    "pose": np.zeros((N_points, 7)),
    "vel": np.zeros((N_points, 6)),
    "acc": np.zeros((N_points, 6)),
    }


    for i in range(N_points):
        s, sdot, sdotdot = cubic_time_scaling(Tf, i*diff)
        waypoints['pose'][i][0:3] = s * pend + (1 - s) * pstart
        waypoints['vel'][i][0:3] = sdot * (pend-pstart)
        waypoints['acc'][i][0:3] = sdotdot * (pend-pstart)

        ## Using SLERP here - https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp --> constant ang vel
        qq = quaternion.slerp_evaluate(quaternion.as_quat_array(qstart), quaternion.as_quat_array(qend), s)
        qq = np.array([qq.w, qq.x, qq.y, qq.z])

        waypoints['pose'][i][3:] = qq

        

    for i in range(N_points-1):
        waypoints['vel'][i][3:] = angular_vel_from_quat(waypoints['pose'][i][3:], waypoints['pose'][i+1][3:], diff)
        waypoints['acc'][i][3:] = (waypoints['vel'][i][3:] - waypoints['vel'][i-1][3:]) / diff
        #print("vel", waypoints['vel'][i][3:])
        #print("acc", waypoints['acc'][i][3:])

    return waypoints







if __name__ == "__main__":

    import matplotlib.pyplot as plt


    ss = np.array([1,2,3,1, 0, 0, 0])
    ee = np.array([7,6,9,0.7071068, 0, 0.7071068, 0])

   
    Np = 20
    wp = straight_line_pose_waypoints(ss, ee, 10, Np)

    plt.plot(wp['acc'][:, 3:])
    print(wp['acc'][:, 3:])
    plt.show()




