#####################################################
## # ## ### < class: Error Propagation  > ### ## # ##
## # ## ### < class: Uncertainty Prop.  > ### ## # ##
## https://github.com/conconga/error_propagation/  ##
#####################################################
## # ## ### < author: Luciano Kruk      > ### ## # ##
#####################################################

import scipy.misc as misc
import math       as m
import numpy      as np

#####################################################

class knumuncert:
    def __init__(self, x, dx):
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
            ret = self.__truediv__(self.__class__(y, 0))
        return ret

    def __itruediv__(self, y): # /=
        q       = self.__truediv__(y)
        self.x  = q.x
        self.dx = q.dx
        return(self)

    #( --- generic functions --- )#
    def function(self, fn):
        q  = fn(self.x)
        dq = self.dx * abs(misc.derivative(fn, self.x))
        return self.__class__(q, dq)

    #( --- miscelaneous --- )#
    def __abs__(self):
        return self.__class__(abs(self.x), self.dx)

    #( --- formatting --- )#
    def __format__(self, fmt):
        return "({{:{:s}}} +- {{:{:s}}})".format(fmt, fmt).format(self.x, self.dx)



#####################################################
if __name__ == "__main__":
    a = knumuncert(1,0.1)
    b = knumuncert(2,0.1)
    c = 3

    print("a = {:1.02f}".format(a))
    print("b = {:1.02f}".format(b))
    print("c = {:1.02f}".format(c))

    print()
    print("a+b = {:1.02f}".format(a+b))
    print("b+a = {:1.02f}".format(b+a))
    print("a+c = {:1.02f}".format(a+c))
    print("c+a = {:1.02f}".format(c+a))

    print()
    print("a-b = {:1.02f}".format(a-b))
    print("b-a = {:1.02f}".format(b-a))
    print("a-c = {:1.02f}".format(a-c))
    print("c-a = {:1.02f}".format(c-a))

    print()
    print("a*b = {:1.02f}".format(a*b))
    print("b*a = {:1.02f}".format(b*a))
    print("a*c = {:1.02f}".format(a*c))
    print("c*a = {:1.02f}".format(c*a))

    print()
    print("a/b = {:1.02f}".format(a/b))
    print("b/a = {:1.02f}".format(b/a))
    print("a/c = {:1.02f}".format(a/c))
    print("c/a = {:1.02f}".format(c/a))

    print()
    print("a**2 = {:1.02f}".format(a.function(lambda x: x**2)))
    print("b**2 = {:1.02f}".format(b.function(lambda x: x**2)))

    print()
    print("a*b/2 = {:1.02f}".format(a*b/2.0))
    print("sin(a*b/2) = {:1.02f}".format((a*b/2.0).function(lambda x:m.sin(x))))

    # operacoes:
    print()
    a += b 
    print("a += b  = {:1.02f}".format(a))
    a -= b
    print("a -= b  = {:1.02f}".format(a))
    a *= b
    print("a *= b  = {:1.02f}".format(a))
    a /= b
    print("a /= b  = {:1.02f}".format(a))

    print()
    print("abs(-a) = {:1.02f}".format(abs(-a)))
    print()

    # nonlinear operations:
    v       = 10
    sigma_v = 0.1
    nb_smps = 10000
    v_smp   = v + (sigma_v * np.random.randn(nb_smps))
    for i in range(2):
        if i == 0:
            fn      = lambda x:x**2
            v_fn    = [fn(i) for i in v_smp]
            v2_fn1  = knumuncert(v, sigma_v).function(fn)
            v2_fn2  = knumuncert(v, sigma_v) * knumuncert(v, sigma_v)
            print('lambda x:x**2')
            print("  from samples, fn(v)     = {:2.2f} +- {:2.2f}".format(fn(v), np.std(v_fn)))
            print("  from er_prop.function() = {:2.2f} +- {:2.2f}".format(v2_fn1.x, v2_fn1.dx))
            print("  from er_prop,   x * x   = {:2.2f} +- {:2.2f}".format(v2_fn2.x, v2_fn2.dx))
            print()

        elif i == 1:
            fn      = lambda x:x**3
            v_fn    = [fn(i) for i in v_smp]
            v2_fn1  = knumuncert(v, sigma_v).function(fn)
            v2_fn2  = knumuncert(v, sigma_v) * knumuncert(v, sigma_v) * knumuncert(v, sigma_v)
            print('lambda x:x**3')
            print("  from samples, fn(v)     = {:2.2f} +- {:2.2f}".format(fn(v), np.std(v_fn)))
            print("  from er_prop.function() = {:2.2f} +- {:2.2f}".format(v2_fn1.x, v2_fn1.dx))
            print("  from er_prop, x * x * x = {:2.2f} +- {:2.2f}".format(v2_fn2.x, v2_fn2.dx))

#####################################################
