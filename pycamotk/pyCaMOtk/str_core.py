from copy import deepcopy
from numpy import mod
try:
    from sympy import sympify, collect_const, simplify
    WITH_SYMPY = True
except ImportError:
    WITH_SYMPY = False

from pyCaMOtk.check import is_type

def cmpr_expr(a, b):
    """
    Test two expressions or iterable of expressions for mathematical
    equivalence using Sympy
    """
    if not is_type(a, str) and not is_type(a, 'iter_of_str'):
        raise TypeError('a must be str or iterable of str')
    if not is_type(b, str) and not is_type(b, 'iter_of_str'):
        raise TypeError('b must be str or iterable of str')
    if is_type(a, 'iter_of_str') and not is_type(b, 'iter_of_str'):
        raise TypeError('a and b must both be str OR both be iterable of str')
    if not is_type(a, 'iter_of_str') and is_type(b, 'iter_of_str'):
        raise TypeError('a and b must both be str OR both be iterable of str')
    if is_type(a, 'iter_of_str') and is_type(b, 'iter_of_str'):
        if len(a) != len(b):
            raise ValueError('a and b must have same length')
        cmpr = [cmpr_expr(a[k], b[k]) for k in range(len(a))]
        return all(cmpr)
    if not WITH_SYMPY: raise ImportError('sympy required')
    asym, bsym = sympify(a), sympify(b)
    return simplify(asym-bsym)==0

def simplify_sympy(expr):
    """
    Simplify expression str or iterable of str using Sympy
    """
    if not WITH_SYMPY: return expr
    if not is_type(expr, str) and not is_type(expr, 'iter_of_str'):
        raise TypeError('expr must be str or iterable of str')
    if is_type(expr, 'iter_of_str'):
        return [simplify_sympy(kexpr) for kexpr in expr]
    #simp_expr = str(collect_const(sympify(expr)))
    simp_expr = str(simplify(expr))
    return 0 if simp_expr == '0' else simp_expr

def paren(s):
    """
    Surround string or list of strings in parentheses
    """
    if is_type(s, 'str'):
        return '({0:s})'.format(s)
    elif hasattr(s, '__iter__'):
        return [paren(ss) for ss in s]
    else:
        raise TypeError('s must be str or list-like of str')
    return s

def smrt_paren(s):
    """
    Surround string in parentheses, only if contents contain operation
    (+, -, *, /, ^)
    """
    if not is_type(s, 'str'):
        raise TypeError('s must be str')
    syms = '+', '-', '*', '/', '^'
    if any([sym in s for sym in syms]):
        return paren(s)
    return s

def strassign(a, b):
    """
    Set string or list of strings (a) to given values (b). If a is list of
    strings and b is not iterable, copy b len(a) times.
    """
    if not is_type(a, 'str') and not is_type(a, 'iter_of_str'):
        raise TypeError('a must be str or iterable of str')
    if type(a) is str:
        return '{0:s} = {1:s}'.format(a, str(b))
    if hasattr(a, '__iter__') and not hasattr(b, '__iter__'):
        expr = [expr for k in range(len(a))]
    return [strassign(a[k], str(b[k])) for k in range(len(a))]

def stradd(a, b, sub=False, simplify=True):
    """
    Return string that represents the sum of a, b. Input may be str or number.
    Option to simpilfy the resulting expression using sympy.
    """
    if not is_type(a, 'str') and not is_type(a, 'number'):
        raise TypeError('a must be str or number')
    if not is_type(b, 'str') and not is_type(b, 'number'):
        raise TypeError('b must be str or number')
    if a == 0:
        if b == 0: return 0
        return str(b) if not sub else '-{0:s}'.format(smrt_paren(str(b)))
    if b == 0:
        return 0 if a == 0  else str(a)
    #a, b = smrt_paren(str(a)), smrt_paren(str(b))
    #expr = '{0:s}+{1:s}'.format(a, b) if not sub else \
    #       '{0:s}-{1:s}'.format(a, b)
    expr = '{0:s}+{1:s}'.format(str(a), str(b)) if not sub else \
           '{0:s}-{1:s}'.format(str(a), smrt_paren(str(b)))
    return simplify_sympy(expr) if simplify else expr

def strsub(a, b, simplify=True):
    """
    Return string that represents the difference of a, b. Input may be str or
    number. Option to simpilfy the resulting expression using sympy.
    """
    if not is_type(a, 'str') and not is_type(a, 'number'):
        raise TypeError('a must be str or number')
    if not is_type(b, 'str') and not is_type(b, 'number'):
        raise TypeError('b must be str or number')
    return stradd(a, b, sub=True, simplify=simplify)

def strmul(a, b, div=False, simplify=True):
    """
    Return string that represents the product of a, b. Input may be str or
    number. Option to simpilfy the resulting expression using sympy.
    """
    if not is_type(a, 'str') and not is_type(a, 'number'):
        raise TypeError('a must be str or number')
    if not is_type(b, 'str') and not is_type(b, 'number'):
        raise TypeError('b must be str or number')
    if div: return strdiv(a, b, simplify=simplify)
    if a == 0 or b == 0: return 0
    if a == 1          : return str(b)
    if b == 1          : return str(a)
    a, b = str(a), str(b)
    if '+' in a or '-' in a:
        a = paren(a)
    if '+' in b or '-' in b:
        b = paren(b)
    expr = '{0:s}*{1:s}'.format(a, b)
    return simplify_sympy(expr) if simplify else expr

def strdiv(a, b, simplify=True):
    """
    Return string that represents the quotient of a, b. Input may be str or
    number. Option to simpilfy the resulting expression using sympy.
    """
    if not is_type(a, 'str') and not is_type(a, 'number'):
        raise TypeError('a must be str or number')
    if not is_type(b, 'str') and not is_type(b, 'number'):
        raise TypeError('b must be str or number')
    if b == 0: raise ValueError('Cannot divide by zero')
    if a == 0: return 0
    if b == 1: return str(a)
    a, b = smrt_paren(str(a)), smrt_paren(str(b))
    expr = '{0:s}/{1:s}'.format(a, b)
    return simplify_sympy(expr) if simplify else expr

def strinv(a, simplify=True):
    """
    Return string that represents the inverse of a. Input may be str or
    number. Option to simpilfy the resulting expression using sympy.
    """
    if not is_type(a, 'str') and not is_type(a, 'number'):
        raise TypeError('a must be str or number')
    expr = strdiv(1, a, simplify=simplify)
    return simplify_sympy(expr) if simplify else expr

def strpow(a, b, simplify=True):
    """
    Return string that represents the a^b. Input may be str or number.
    Option to simplify the resulting expression using sympy.
    """
    if not is_type(a, 'str') and not is_type(a, 'number'):
        raise TypeError('a must be str or number')
    if not is_type(b, 'str') and not is_type(b, 'number'):
        raise TypeError('b must be str or number')
    if b == 0: return 1
    if b == 1: return str(a)
    expr = '{0:s}^{1:s}'.format(smrt_paren(str(a)), smrt_paren(str(b)))
    return simplify_sympy(expr) if simplify else expr

def strunop(a, op=None, simplify=True):
    """
    Return string that represents the unary operation to a, i.e., op(a). Input
    may be str or number. Option to simplify the resulting expression using
    sympy.
    """
    if not is_type(a, 'str') and not is_type(a, 'number'):
        raise TypeError('a must be str or number')
    if not is_type(op, 'str'):
        raise TypeError('op must be str')
    if op == 'sqrt':
        if a == 0: return 0
        if a == 1: return 1
    if op == 'exp':
        if a == 0: return 1
    if op == 'abs':
        if a == 0: return 0
        if a == -1 or a == 1: return 1
    expr = '{1:s}({0:s})'.format(a, op)
    return simplify_sympy(expr) if simplify else expr

def strheavi(a):
    """
    Return string that represents Heavidside(a). Input may be str or number.
    """
    if not is_type(a, 'str') and not is_type(a, 'number'):
        raise TypeError('a must be str or number')
    return 'Heaviside('+str(a)+')'

def strtern(cond, a, b):
    """
    Return string that represents ternary operator (cond ? a : b) using the
    Heaviside function. Instead of cond being the traditional boolean (T/F),
    it is a float that is positive (T) or negative (F). Input may be str or
    number.
    """
    return stradd(strmul(a, strheavi(cond)),
                  strmul(b, stradd(1, strheavi(cond), True)))

def strmax2(a, b):
    """
    Return string that represents max(a, b) using the Heaviside function.
    Input may be str or number.
    """
    return stradd(strmul(a, strheavi(stradd(a, b, True))),
                  strmul(b, strheavi(stradd(b, a, True))))

def strmax(v):
    """
    Return string that represents max(v[0], ..., v[-1]) using the
    Heaviside function. Input may be str or number.
    """
    N, expr = len(v), 0
    for k in range(N):
        a, s = v[k], v[k]
        for j in range(N):
            if j==k: continue
            b = v[j]
            s = strmul(s, strheavi(strsub(a, b)))
        expr = stradd(expr, s)
    return expr

def format_strblk(lines, lvl=0, *args):
    """
    Convert nested iterables of str to a formatted string where an
    equal number of indents is applied to all str at the same nesting level

    Input arguments
    ---------------
    lines : nested iterable of str
      All entries that are str are in "current" level and all entries that
      are iterable correspond to deeper levels
    lvl : int
      Indentation level for current block

    Return value
    ------------
    s : str
      Formatted string

    Example
    -------
    >> lines = ['int main()', '{', ['std::cout << "Hello World";'], '}']
    >> format_strblk(lines, 0)
    # 'int main()\n{\n  std::cout << "Hello World";\n}\n'
    """
    if not is_type(lines, 'iter') and not is_type(lines, 'str'):
        raise TypeError('lines must be str or nested iterable of str')
    if not is_type(lvl, 'int'):
        raise TypeError('lvl must be int')
    s = ''
    for line in lines:
        if is_type(line, 'str'):
            if len(args) > 0:
                s += (2*lvl)*' ' + line.format(*args) + '\n'
            else:
                s += (2*lvl)*' ' + line + '\n'
        else:
            s += format_strblk(line, lvl+1, *args)
    return s
