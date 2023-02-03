from numba import njit,jit, prange
import numpy as np

def transform_is_valid(t, tolerance=1e-3):
    """ Check if array is a valid transform.
    You can refer to the lecture notes to 
    see how to check if a matrix is a valid
    transform. 
    
    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.
    """

    rm = t[:3,:3]
    bool_1 = 1
    try:
        
        inv_rm = np.linalg.inv(rm)
        if np.linalg.det(rm) > 1+tolerance or np.linalg.det(rm) < 1-tolerance or t[3,0] !=0 or t[3,1] !=0 or t[3,2] !=0 or t[3,3] !=1:
            bool_1 = 0

    except:
        bool_1 = 0

    return bool_1


def transform_concat(t1, t2):
    """ Concatenate two transforms. Hint: 
        use numpy matrix multiplication. 

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2.
    """
    buffer = 1

    if transform_is_valid(t1)==0:
        buffer = 0
        raise ValueError('t1 is invalid.')
    if transform_is_valid(t2)==0:
        buffer = 0
        raise ValueError('t2 is invalid.')
    
    if buffer != 0:
        rm4 = t1@t2
    return rm4

def transform_point3s(t, ps):
    """ Transfrom a list of 3D points
    from one coordinate frame to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    if transform_is_valid(t)==0:
        raise ValueError('t is not a valid transform.')
    
    for i in range(len(ps)):
        if len(ps[i]) != 3:
            raise ValueError('ps is not a valid transform.') 
        
    supply = np.ones(len(ps))
    supply = np.expand_dims(supply, axis=0)
    ps = np.array(ps)

    ps_changed = t@(np.concatenate((ps.T,supply),axis=0))

    return np.delete(ps_changed,3,0).T



def transform_inverse(t):
    """Find the inverse of the transfom. Hint:
        use Numpy's linear algebra native methods. 

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    if transform_is_valid(t)==1:
        t = np.linalg.inv(t)
    return t

