import numpy as np
import math


def sph_to_cart(epsilon, alpha, r):
    """
    Transform sensor readings to Cartesian coordinates in the sensor
    frame. The values of epsilon and alpha are given in radians, while
    r is in metres. Epsilon is the elevation angle and alpha is the
    azimuth angle (i.e., in the x,y plane).
    """
    P = np.zeros(3)  # Position vector
    print(P)
    # Your code here
    P[2] = np.sin(epsilon)*r
    P[0] = np.cos(alpha)*np.cos(epsilon)*r
    P[1] = np.sin(alpha)*np.cos(epsilon)*r

    return P


def cart_to_sph(x1, y1, z1):

    # Your code here
    epsilon = math.atan(z1/math.sqrt(y1**2+x1**2))
    alpha = math.acos(x1/math.sqrt(y1**2+x1**2))
    r = math.pow(x1**2+y1**2+z1**2, 1/2)

    return [epsilon, alpha, r]


print(cart_to_sph(1, 1, 1))


def quaternion(direction, angle):

    direction = np.array(direction)
    direction_n = direction / \
        (math.sqrt(direction[0]**2+direction[1]**2+direction[2]**2))
    direction_n = direction_n*math.sin(angle)
    direction_n = np.append(direction_n, np.array([math.cos(angle)]))

    return direction_n


print(quaternion([0, 0.5, 0], 180/180*np.pi))


# def quaternion_rotation_matrix(Q):
#     """
#     Covert a quaternion into a full three-dimensional rotation matrix.

#     Input
#     :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

#     Output
#     :return: A 3x3 element matrix representing the full 3D rotation matrix.
#              This rotation matrix converts a point in the local reference
#              frame to a point in the global reference frame.
#     """
#     # Extract the values from Q
#     q0 = Q[0]
#     q1 = Q[1]
#     q2 = Q[2]
#     q3 = Q[3]

#     # First row of the rotation matrix
#     r00 = 2 * (q0 * q0 + q1 * q1) - 1
#     r01 = 2 * (q1 * q2 - q0 * q3)
#     r02 = 2 * (q1 * q3 + q0 * q2)

#     # Second row of the rotation matrix
#     r10 = 2 * (q1 * q2 + q0 * q3)
#     r11 = 2 * (q0 * q0 + q2 * q2) - 1
#     r12 = 2 * (q2 * q3 - q0 * q1)

#     # Third row of the rotation matrix
#     r20 = 2 * (q1 * q3 - q0 * q2)
#     r21 = 2 * (q2 * q3 + q0 * q1)
#     r22 = 2 * (q0 * q0 + q3 * q3) - 1

#     # 3x3 rotation matrix
#     rot_matrix = np.array([[r00, r01, r02],
#                            [r10, r11, r12],
#                            [r20, r21, r22]])

#     return rot_matrix
