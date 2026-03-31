#####################################################
## # ## ### < class: Error Propagation  > ### ## # ##
## # ## ### < class: Uncertainty Prop.  > ### ## # ##
## https://github.com/conconga/error_propagation/  ##
#####################################################
## # ## ### < author: Luciano Kruk      > ### ## # ##
#####################################################

import math       as m
import numpy      as np
from scipy.differentiate import derivative

#####################################################

class knumuncert:
    def __init__(self, x, dx):
        assert dx is not None
        self.x  = x
        self.dx = dx

    def __repr__(self):
        return "{:1.1e}+-{:1.1e}".format(self.x, self.dx)

    #( --- sum --- )#
    def __add__(self, y):
        if isinstance(y, self.__class__):
            ret = self.__class__(self.x + y.x, m.sqrt(sum([i**2.0 for i in [self.dx, y.dx]])))
        else:
            ret = self.__class__(self.x + y, self.dx)
        return ret

    def __radd__(self, y):
        return self.__add__(y)

    def __iadd__(self, y): # +=
        q       = self.__add__(y)
        self.x  = q.x
        self.dx = q.dx
        return(self)

    #( --- negative signal --- )#
    def __neg__(self):
        return self.__class__(-self.x, self.dx)

    #( --- difference --- )#
    def __sub__(self, y):
        if isinstance(y, self.__class__):
            ret = self.__class__(self.x - y.x, m.sqrt(sum([i**2.0 for i in [self.dx, y.dx]])))
        else:
            ret = self.__class__(self.x - y, self.dx)
        return ret

    def __rsub__(self, y):
        return y+self.__class__(-self.x, self.dx)

    def __isub__(self, y): # -=
        q       = self.__sub__(y)
        self.x  = q.x
        self.dx = q.dx
        return(self)

    #( --- multiplication --- )#
    def __mul__(self, y):
        if isinstance(y, self.__class__):
            q   = self.x * y.x
            dq  = abs(q) * m.sqrt(sum([i**2.0 for i in [self.dx/self.x, y.dx/y.x]]))
            ret = self.__class__(q, dq)
        else:
            ret = self.__mul__(self.__class__(y,0))
        return ret

    def __rmul__(self, y):
        return self.__mul__(y)

    def __imul__(self, y): # *=
        q       = self.__mul__(y)
        self.x  = q.x
        self.dx = q.dx
        return(self)

    #( --- division --- )#
    def __truediv__(self, y):
        if isinstance(y, self.__class__):
            q   = self.x / y.x
            dq  = abs(q) * m.sqrt(sum([i**2.0 for i in [self.dx/self.x, y.dx/y.x]]))
            ret = self.__class__(q, dq)
        else:
            ret = self.__truediv__(self.__class__(y, 0))
        return ret

    def __rtruediv__(self, y):
        if isinstance(y, self.__class__):
            q   = y.x / self.x
            dq  = abs(q) * m.sqrt(sum([i**2.0 for i in [self.dx/self.x, y.dx/y.x]]))
            ret = self.__class__(q, dq)
        else:
            ret = self.__class__(y,0) / self
        return ret

    def __itruediv__(self, y): # /=
        q       = self.__truediv__(y)
        self.x  = q.x
        self.dx = q.dx
        return(self)

    #( --- generic functions --- )#
    def function(self, fn):
        """
        The function 'fn' shall support its calculation with arrays. The numerical
        derivative of 'fn' will call it with arrays in the input.
        """
        q   = fn(self.x)
        ret = derivative(fn, self.x)
        if ret['success']:
            dq = self.dx * abs(ret['df'])
        else:
            raise ValueError('the derivative did not converge')

        return self.__class__(q, dq)

    #( --- miscelaneous --- )#
    def __abs__(self):
        return self.__class__(abs(self.x), self.dx)

    #( --- formatting --- )#
    def __format__(self, fmt):
        return "({{:{:s}}} +- {{:{:s}}})".format(fmt, fmt).format(self.x, self.dx)

    #( --- copy --- )#
    def copy(self):
        return self.__class__(self.x, self.dx)


#####################################################
