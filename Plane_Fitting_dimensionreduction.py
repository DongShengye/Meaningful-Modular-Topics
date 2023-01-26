import numpy as np


def estimate_params(p):
    """
    Estimate parameters from sensor readings in the Cartesian frame.
    Each row in the P matrix contains a single 3D point measurement;
    the matrix P has size n x 3 (for n points). The format is:

    P = [[x1, y1, z1],
         [x2, x2, z2], ...]

    where all coordinate values are in metres. Three parameters are
    required to fit the plane, a, b, and c, according to the equation

    z = a + bx + cy

    The function should return the parameters as a NumPy array of size
    three, in the order [a, b, c].
    """
    param_est = np.zeros(3)

    # Your code here
    a1 = np.ones(len(p)).T
    a2 = p[:, 0]
    a3 = p[:, 1]
    Y = p[:, 2]
    #print(a1, a2, a3)
    A = np.c_[a1, a2, a3]
    print("A is:", A)
    if np.linalg.matrix_rank(A.T@A) >= np.shape(A.T@A)[0]:
        Xls = np.linalg.inv(A.T@A)@A.T@Y
    else:
        # pseudo inverse if function is not full rank
        Xls = np.linalg.pinv(A.T@A)@A.T@Y
    param_est[0] = Xls[0]
    param_est[1] = Xls[1]
    param_est[2] = Xls[2]
    return param_est


test = np.array([[1, 0, 0], [1, 1, 0], [0.2, 0.8, 0]])

print("The optimal parameters are:", estimate_params(test))


# def sph_to_cart(epsilon, alpha, r):
#     """
#     Transform sensor readings to Cartesian coordinates in the sensor
#     frame. The values of epsilon and alpha are given in radians, while
#     r is in metres. Epsilon is the elevation angle and alpha is the
#     azimuth angle (i.e., in the x,y plane).
#     """
#     P = np.zeros(3)  # Position vector
#     print(P)
#     # Your code here
#     P[2] = np.sin(epsilon)*r
#     P[0] = np.cos(alpha)*np.cos(epsilon)*r
#     P[1] = np.sin(alpha)*np.cos(epsilon)*r

#     return P


#print(sph_to_cart(5/180*np.pi, 10/180*np.pi, 4))
