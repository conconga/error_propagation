#####################################################
## # ## ### < class: Error Propagation  > ### ## # ##
## # ## ### < class: Uncertainty Prop.  > ### ## # ##
## https://github.com/conconga/error_propagation/  ##
#####################################################
## # ## ### < author: Luciano Kruk      > ### ## # ##
#####################################################

import scipy.misc as misc
import math       as m

#####################################################

class CNUMINCERT:
    def __init__(self, x, dx):
        self.x  = x
        self.dx = dx

    def __repr__(self):
        return "{:1.1e}+-{:1.1e}".format(self.x, self.dx)

    #( --- sum --- )#
    def __add__(self, y):
        if isinstance(y, CNUMINCERT):
            ret = CNUMINCERT(self.x + y.x, m.sqrt(sum([i**2.0 for i in [self.dx, y.dx]])))
        else:
            ret = CNUMINCERT(self.x + y, self.dx)
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
        return CNUMINCERT(-self.x, self.dx)

    #( --- difference --- )#
    def __sub__(self, y):
        if isinstance(y, CNUMINCERT):
            ret = CNUMINCERT(self.x - y.x, m.sqrt(sum([i**2.0 for i in [self.dx, y.dx]])))
        else:
            ret = CNUMINCERT(self.x - y, self.dx)
        return ret

    def __rsub__(self, y):
        return y+CNUMINCERT(-self.x, self.dx)

    def __isub__(self, y): # -=
        q       = self.__sub__(y)
        self.x  = q.x
        self.dx = q.dx
        return(self)

    #( --- multiplication --- )#
    def __mul__(self, y):
        if isinstance(y, CNUMINCERT):
            q   = self.x * y.x
            dq  = abs(q) * m.sqrt(sum([i**2.0 for i in [self.dx/self.x, y.dx/y.x]]))
            ret = CNUMINCERT(q, dq)
        else:
            ret = self.__mul__(CNUMINCERT(y,0))
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
        if isinstance(y, CNUMINCERT):
            q   = self.x / y.x
            dq  = abs(q) * m.sqrt(sum([i**2.0 for i in [self.dx/self.x, y.dx/y.x]]))
            ret = CNUMINCERT(q, dq)
        else:
            ret = self.__truediv__(CNUMINCERT(y, 0))
        return ret

    def __rtruediv__(self, y):
        if isinstance(y, CNUMINCERT):
            q   = y.x / self.x
            dq  = abs(q) * m.sqrt(sum([i**2.0 for i in [self.dx/self.x, y.dx/y.x]]))
            ret = CNUMINCERT(q, dq)
        else:
            ret = self.__truediv__(CNUMINCERT(y, 0))
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
        return CNUMINCERT(q, dq)

    #( --- miscelaneous --- )#
    def __abs__(self):
        return CNUMINCERT(abs(self.x), self.dx)

    #( --- formatting --- )#
    def __format__(self, fmt):
        return "({{:{:s}}} +- {{:{:s}}})".format(fmt, fmt).format(self.x, self.dx)



#####################################################
if __name__ == "__main__":
    a = CNUMINCERT(1,0.1)
    b = CNUMINCERT(2,0.1)
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


#####################################################
