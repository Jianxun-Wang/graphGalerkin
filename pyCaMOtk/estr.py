import numpy as np

from pyCaMOtk.check import is_type
from pyCaMOtk.str_core import stradd, strsub, strmul, strdiv, \
                              strpow, strunop

class estr(str):
    """
    Subclass of str that overwrite traditional string operations, e.g.,
    "+" for concatenation, with string operations in str_core
    """
    def __init__(self, s):
        self.s = str(s)
    @staticmethod
    def exp(s):
        return estr(strunop(s, 'exp'))
    @staticmethod
    def sqrt(s):
        return estr(strunop(s, 'sqrt'))
    def __add__(self, x):
        if is_type(x, np.ndarray) and isinstance(x[0], estr):
            return x+estr(self.s)
        return estr(stradd(self.s, x))
    def __sub__(self, x):
        if is_type(x, np.ndarray) and isinstance(x[0], estr):
            return -(x-estr(self.s))
        return estr(strsub(self.s, x))
    def __mul__(self, x):
        if is_type(x, np.ndarray) and isinstance(x[0], estr):
            return x*estr(self.s)
        return estr(strmul(self.s, x))
    def __div__(self, x):
        return estr(strdiv(self.s, x))
    def __truediv__(self, x):
        return estr(strdiv(self.s, x))
    def __floordiv__(self, x):
        return self.__truediv__(x)
    def __radd__(self, x):
        return self.__add__(x)
    def __rsub__(self, x):
        return -self.__sub__(x)
    def __rmul__(self, x):
        return self.__mul__(x)
    def __rdiv__(self, x):
        return estr(strdiv(x, self.s))
    def __rtruediv__(self, x):
        return estr(strdiv(x, self.s))
    def __rfloordiv__(self, x):
        return self.__rtruediv__(x)
    def __pow__(self, a):
        return estr(strpow(self.s, a))
    def __iadd__(self, x):
        self.s = stradd(self.s, x)
        return estr(self.s)
    def __isub__(self, x):
        self.s = stradd(self.s, x)
        return estr(self.s)
    def __imul__(self, x):
        self.s = strmul(self.s, x)
        return estr(self.s)
    def __itruediv__(self, x):
        self.s = strdiv(self.s, x)
        return estr(self.s)
    def __ifloordiv__(self, x):
        self.__itruediv__(x)
    def __ipow__(self, a):
        self.s = strpow(self.s, a)
        return estr(self.s)
    def __neg__(self):
        return estr(strmul(-1, self.s))
        #self.s = strmul(-1, self.s)
        #return estr(self.s)
    def __pos__(self):
        return estr(self.s)
    def __abs__(self):
        return estr(strunop(self.s, 'abs'))
        #self.s = strunop(self.s, 'abs')
        #return estr(self.s)
    def __inv__(self):
        return estr(strdiv(1, self.s))
        #self.s = strdiv(1, self.s)
        #return estr(self.s)
    def __str__(self):
        return self.s
    def __repr__(self):
        return self.s

def create_ndarray_of_estr(dims, strlst):
    """
    Create Numpy ndarray (type: estr, shape: dims) from iterable of
    strings (strlst).
    """
    if not is_type(strlst, 'iter_of_str_number'):
        raise TypeError('strlst must be iterable of str')
    estrlst = [(s if isinstance(s, estr) else estr(s)) for s in strlst]
    M = np.array(estrlst, dtype=estr, order='F')
    M = M.reshape(dims, order='F')
    return M
