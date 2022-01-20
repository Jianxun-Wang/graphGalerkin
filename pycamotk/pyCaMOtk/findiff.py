
from pyCaMOtk.check import is_type

def fd_1deriv_info(scheme):
    """
    First-order finite difference coefficients alpha, beta, gamma such that:

        f'(x) = (alpha/eps)*(sum_k beta[k]*f(x+gamma[k]*eps)) + O(eps^p)

    Input arguments
    ---------------
    scheme : str
      Name of finite difference scheme ('fd1', 'bd1', 'cd2',
                                        'cd4', 'cd6', 'cd8')

    Return values
    -------------
    alpha, beta, gamma : float, iterable of float, iterable of float
      Finite difference coefficients

    Example
    -------
    >> alpha, beta, gamma = fd_1deriv_info('cd2')
    # alpha, beta, gamma = 0.5, [-1.0, 1.0], [-1.0, 1.0]
    """
    if not is_type(scheme, 'str'):
        raise TypeError('scheme must be str')
    perm = None
    if scheme == 'fd1':
        alpha = 1.0
        beta = [-1.0, 1.0]
        gamma = [0.0, 1.0]
    elif scheme == 'bd2':
        alpha = 1.0
        beta = [-1.0, 1.0]
        gamma = [-1.0, 0.0]
    elif scheme == 'cd2':
        alpha = 0.5
        beta = [-1.0, 1.0]
        gamma = [-1.0, 1.0]
    elif scheme == 'cd4':
        alpha = 1.0/12.0
        beta = [1.0, -8.0, 8.0, -1.0]
        gamma = [-2.0, -1.0, 1.0, 2.0]
        perm = [3, 0, 2, 1]
    elif scheme == 'cd6':
        alpha = 1.0/60.0
        beta = [-1.0, 9.0, -45.0, 45.0, -9.0, 1.0]
        gamma = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        perm = [5, 0, 4, 1, 3, 2]
    elif scheme == 'cd8':
        alpha = 1.0/840.0
        beta = [3.0, -32.0, 168.0, -672.0, 672.0, -168.0, 32.0, -3.0]
        gamma = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]
        perm = [7, 0, 6, 1, 5, 2, 4, 3]
    else:
        raise ValueError('Finite difference scheme not supported')
    if perm is not None:
        beta, gamma = [beta[k] for k in perm], [gamma[k] for k in perm]
    return alpha, beta, gamma

def fd_1deriv_dir(f, eps, x, dx, scheme):
    """
    First-order directional derivative of f about x in the direction dx

       Df(x; dx) = (alpha/eps)*(sum_k beta[k]*f(x+gamma[k]*eps*dx)) + O(eps^p)

    Input arguments
    ---------------
    f : function
    eps : float
      Finite difference step
    x : iterable of float
      Point about which to compute directional derivative
    dx : iterable of float
      Direction to compute derivative
    scheme : str
      Name of finite difference scheme (see fd_1deriv_info(...))

    Return values
    -------------
    df : iterable of float
      Finite difference approximation of directional derivative

    Example
    -------
    >> f = lambda x: np.array([x[0]**2, x[1], x[0]*x[1]])
    >> x, dx = np.array([1.0, 2.0]), np.array([2.0, 1.0])
    >> fd_1deriv_dir(f, 1.0e-6, x, dx, 'cd2')
    # np.array([4.0, 1.0, 5.0])
    """
    if not is_type(eps, 'number'):
        raise TypeError('eps must be number')
    if not is_type(x, 'ndarray_of_number'):
        raise TypeError('x must be ndarray of number')
    if not is_type(dx, 'ndarray_of_number'):
        raise TypeError('dx must be ndarray of number')
    if not is_type(scheme, 'str'):
        raise TypeError('scheme must be str')
    alpha, beta, gamma = fd_1deriv_info(scheme)
    df = beta[0]*f(x+gamma[0]*eps*dx)
    for k in range(1, len(gamma)):
        df += beta[k]*f(x+gamma[k]*eps*dx)
    df *= alpha/eps
    return df

def fd_1deriv_jac(f, eps, x, scheme, df):
    """
    Jacobian of f about x

     Df(x)[..., j] = (alpha/eps)*(sum_k beta[k]*f(x+gamma[k]*eps*ej)) + O(eps^p)

    Input arguments
    ---------------
    f : function
    eps : float
      Finite difference step
    x : ndarray of float
      Point about which to compute directional derivative
    scheme : str
      Name of finite difference scheme (see fd_1deriv_info(...))

    Return values
    -------------
    df : ndarray of float
      Finite difference approximation of Jacobian

    Example
    -------
    >> f = lambda x: np.array([x[0]**2, x[1], x[0]*x[1]])
    >> df = np.zeros((3, 2), dtype=float, order='F')
    >> x = np.array([1.0, 2.0])
    >> fd_1deriv_jac(f, 1.0e-6, x, 'cd2', df)
    # np.array([[2.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    """
    if not is_type(eps, 'number'):
        raise TypeError('eps must be number')
    if not is_type(x, 'ndarray_of_number'):
        raise TypeError('x must be ndarray of number')
    if not is_type(scheme, 'str'):
        raise TypeError('scheme must be str')
    if not is_type(df, 'ndarray_of_number'):
        raise TypeError('df must be ndarray of number')
    nx = x.size
    for k in range(nx):
        ek = 0*x
        ek[k] = 1.0
        df[..., k] = fd_1deriv_dir(f, eps, x, ek, scheme)
    return df

if __name__ == '__main__':
    import numpy as np
    f = lambda x: np.array([x[0]**2, x[1], x[0]*x[1]])
    x, dx = np.array([1.0, 2.0]), np.array([2.0, 1.0])
    print fd_1deriv_dir(f, 1.0e-6, x, dx, 'cd2')
    df = np.zeros((3, 2), dtype=float, order='F')
    print fd_1deriv_jac(f, 1.0e-6, x, 'cd2', df)
