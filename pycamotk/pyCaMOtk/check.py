import numpy as np
from numbers import Number, Integral

def is_within_bounds(dims, mltidx):
    """
    Check if mltidx is a valid index into an array with dimensions dims
    (0-based)

    Input arguments
    ---------------
    dims : int or iterable of int

    mltidx : int or iterable of int

    Return value
    ------------
    bool indicating whether mltidx is valid index for array with dims

    Examples
    --------
    >> is_within_bounds(10,  2) # True
    >> is_within_bounds(10, -2) # True
    >> is_within_bounds(10, 11) # False
    >> is_within_bounds((2, 3, 4), ( 0, 1,  2)) # True
    >> is_within_bounds((2, 3, 4), (-1, 0, -2)) # True
    >> is_within_bounds((2, 3, 4), ( 0, 1,  4)) # False
    """
    if is_type(dims, 'int') and is_type(mltidx, 'int'):
        return mltidx<dims or mltidx>-dims-1
    elif is_type(dims, 'iter_of_int') and is_type(mltidx, 'iter_of_int'):
        if not len(dims) == len(mltidx):
            raise ValueError('dims and mltidx must be same size')
        ndim = len(dims)
        sztst = [mltidx[k]<dims[k] or mltidx[k]>-dims[k]-1 for k in range(ndim)]
        return all(sztst)
    else:
        raise TypeError('Invalid input: dims/mltidx must be comparable')

def is_type(obj, type_or_str):
    """
    Check type of obj against type_or_str

    Input arguments
    ---------------
    obj : type

    type_or_str : type or str
        Admissible strings: str, iter, iter_of_*, int,
                            number, ndarray, ndarray_of_*

    Return value
    ------------
    bool indicating object is of type type_or_str

    Examples
    --------
    >> is_type('s', 'str') # True
    >> is_type(['s', 'w'], 'iter_of_str') # True
    >> is_type([0, 1], 'iter_of_int') # True
    >> is_type([0.3, 1.2], 'iter_of_number') # True
    """
    if isinstance(type_or_str, type):
        return isinstance(obj, type_or_str)
    if isinstance(type_or_str, str):
        if type_or_str == 'str':
            return isinstance(obj, str)
        if type_or_str == 'iter':
            return hasattr(obj, '__iter__')
        if 'iter_of' in type_or_str:
            if not hasattr(obj, '__iter__'): return False
            if 'iter_of' in type_or_str[8:]:
                all_type = [type_or_str[8:]]
            else:
                all_type = type_or_str[8:].split('_')
            for kobj in obj:
                is_any_type = any([is_type(kobj, jtype) for jtype in all_type])
                if not is_any_type: return False
            return True
        if type_or_str == 'int':
            return isinstance(obj, Integral)
        if type_or_str == 'number':
            return isinstance(obj, Number)
        if type_or_str == 'ndarray':
            return isinstance(obj, np.ndarray)
        if 'ndarray_of' in type_or_str:
            if not isinstance(obj, np.ndarray): return False
            if len(obj) == 0: 
                raise ValueError('Operation not valid for ndarray of length 0')
            return is_type(obj.flatten('F')[0], type_or_str[11:])
    raise ValueError('type_or_str must be type or str')
