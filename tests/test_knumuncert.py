#####################################################
## # ## ### < class: Error Propagation  > ### ## # ##
## # ## ### < class: Uncertainty Prop.  > ### ## # ##
## https://github.com/conconga/error_propagation/  ##
#####################################################
## # ## ### < author: Luciano Kruk      > ### ## # ##
#####################################################

import sys    
print(f"** __name__    = {__name__}")
print(f"** __package__ = {__package__}")
print(f"** sys.path[0] = {sys.path[0]}")

from knumuncert import *
import pytest
import numpy as np

if False:
    import pudb
    pudb.set_trace()

#####################################################
class TestClass_knumuncert:

    @pytest.fixture
    def abc(self):
        a = knumuncert(1, 0.1)
        b = knumuncert(2, 0.1)
        c = 3.0
        return { 'a': a, 'b': b, 'c': c }

    #----------------------------------------#
    def test_creating(self):
        a = knumuncert(1,0.1)
        b = knumuncert(2,0.1)

        assert a.x  == 1
        assert a.dx == 0.1
        assert b.x  == 2
        assert b.dx == 0.1

    #----------------------------------------#
    @pytest.mark.parametrize(
            "a,da,b,db, operation, c,dc", [
                (1, 0.1,  2, 0.1,  'sum', 3, 0.14),
                (2, 0.1,  1, 0.1,  'sum', 3, 0.14),
                (1, 0.1,  3, None, 'sum', 4, 0.10),
                (3, None, 1, 0.1,  'sum', 4, 0.10),

                (1, 0.1,  2, 0.1,  'sub', -1, 0.14),
                (2, 0.1,  1, 0.1,  'sub',  1, 0.14),
                (1, 0.1,  3, None, 'sub', -2, 0.10),
                (3, None, 1, 0.1,  'sub',  2, 0.10),

                (1, 0.1,  2, 0.1,  'mult', 2, 0.22),
                (2, 0.1,  1, 0.1,  'mult', 2, 0.22),
                (1, 0.1,  3, None, 'mult', 3, 0.30),
                (3, None, 1, 0.1,  'mult', 3, 0.30),

                (1, 0.1,  2, 0.1,  'div', 0.5 , 0.06),
                (2, 0.1,  1, 0.1,  'div',   2 , 0.22),
                (1, 0.1,  3, None, 'div', 0.33, 0.03),
                (3, None, 1, 0.1,  'div',   3 , 0.30),
    ])


    def test_arithmetics(self, a, da, b, db, operation, c, dc):
        val_a = a if da is None else knumuncert(a, da)
        val_b = b if db is None else knumuncert(b, db)

        if operation == 'sum':
            res     = val_a + val_b
            op_symb = "+"
        elif operation == 'sub':
            res     = val_a - val_b
            op_symb = "-"
        elif operation == "mult":
            res     = val_a * val_b
            op_symb = "*"
        elif operation == "div":
            res     = val_a / val_b
            op_symb = "/"

        print()
        print(f'{val_a:2.2f} {op_symb} {val_b:2.2f}  =  {res:2.2f} (expected={knumuncert(c,dc):2.2f})')

        assert abs(res.x - c) < 1e-2
        assert abs(res.dx - dc) < 1e-2

    #----------------------------------------#
    def test_power2(self, abc):

        res = abc['a'].function( lambda x: x**2 )
        print()
        print(f'{abc["a"]:2.2f}**2 = {res:2.2f} (expected={knumuncert(1,0.2):2.2f})')
        assert abs(res.x - 1.0) < 1e-2
        assert abs(res.dx - 0.2) < 1e-2

        res = abc['b'].function( lambda x: x**2 )
        print()
        print(f'{abc["b"]:2.2f}**2 = {res:2.2f} (expected={knumuncert(4,0.4):2.2f})')
        assert abs(res.x - 4.0) < 1e-2
        assert abs(res.dx - 0.4) < 1e-2

    #----------------------------------------#
    def test_complex_1(self, abc):

        res = abc['a'] * abc['b'] / 2
        print(f'\n{abc["a"]:2.2f} * {abc["b"]:2.2f} / 2.0 = {res:2.2f} (expected={knumuncert(1.0, 0.11):2.2f})')
        assert abs(res.x - 1.0) < 1e-2
        assert abs(res.dx - 0.11) < 1e-2

    #----------------------------------------#
    def test_complex_2(self, abc):

        res = (abc['a'] * abc['b'] / 2).function( lambda x: np.sin(x) )
        print(f'\nsin({abc["a"]:2.2f} * {abc["b"]:2.2f} / 2.0) = {res:2.2f} (expected={knumuncert(0.84, 0.06):2.2f})')
        assert abs(res.x - 0.84) < 1e-2
        assert abs(res.dx - 0.06) < 1e-2

    #----------------------------------------#
    def test_copy(self, abc):
        copy = abc['a'].copy()
        assert id(copy) != id(abc['a'])
        assert copy.x == abc['a'].x
        assert copy.dx == abc['a'].dx

    #----------------------------------------#
    def test_iadd(self, abc):

        gab = abc['a'] + abc['b']

        res = abc['a'].copy()
        res += abc['b']

        print(f'\n{abc["a"]:2.2f} += {abc["b"]:2.2f} = {res:2.2f} (expected={gab:2.2f})')
        assert abs(res.x - gab.x) < 1e-4
        assert abs(res.dx - gab.dx) < 1e-4

    #----------------------------------------#
    def test_isub(self, abc):

        gab = abc['a'] - abc['b']

        res = abc['a'].copy()
        res -= abc['b']

        print(f'\n{abc["a"]:2.2f} -= {abc["b"]:2.2f} = {res:2.2f} (expected={gab:2.2f})')
        assert abs(res.x - gab.x) < 1e-4
        assert abs(res.dx - gab.dx) < 1e-4

    #----------------------------------------#
    def test_imul(self, abc):

        gab = abc['a'] * abc['b']

        res = abc['a'].copy()
        res *= abc['b']

        print(f'\n{abc["a"]:2.2f} *= {abc["b"]:2.2f} = {res:2.2f} (expected={gab:2.2f})')
        assert abs(res.x - gab.x) < 1e-4
        assert abs(res.dx - gab.dx) < 1e-4

    #----------------------------------------#
    def test_itruediv(self, abc):

        gab = abc['a'] / abc['b']

        res = abc['a'].copy()
        res /= abc['b']

        print(f'\n{abc["a"]:2.2f} /= {abc["b"]:2.2f} = {res:2.2f} (expected={gab:2.2f})')
        assert abs(res.x - gab.x) < 1e-4
        assert abs(res.dx - gab.dx) < 1e-4

    #----------------------------------------#
    def test_neg(self, abc):

        gab = knumuncert( -abc['a'].x, abc['a'].dx )
        res = -abc['a']

        print(f'\nNeg({abc["a"]:2.2f}) = {res:2.2f} (expected={gab:2.2f})')
        assert abs(res.x - gab.x) < 1e-4
        assert abs(res.dx - gab.dx) < 1e-4

    #----------------------------------------#
    def test_abs(self, abc):

        gab = abc["a"].copy()
        res = abs(-abc['a'])

        print(f'\nabs(-{abc["a"]:2.2f}) = {res:2.2f} (expected={gab:2.2f})')
        assert abs(res.x - gab.x) < 1e-4
        assert abs(res.dx - gab.dx) < 1e-4

    #----------------------------------------#
    def test_nonlinear_x2(self):
        """
        nonlinear operations (for the README.md):
        """

        # ground truth:
        val         = 10.0
        val_std     =  0.1
        # simulation: draw nb_samples with mean=val and std=val_std:
        nb_samples  = 10000
        val_samples = val + (val_std * np.random.randn(nb_samples))
        # nonlinear function:
        fn          = lambda x:x**2
        # alternative 1: samples through the nonlinear function:
        y_samples   = [fn(i) for i in val_samples]
        # alternative 2: using knumuncert.function() to propagate the nonlinearity:
        y_function  = knumuncert(val, val_std).function(fn)
        # alternative 3: direct product of two values with uncertainty:
        y_prod      = knumuncert(val, val_std) * knumuncert(val, val_std)
        # printing the results:
        gab = knumuncert(fn(val), np.std(y_samples))
        print(f'\nFunction: y = x^2')
        print(f'  1) from {nb_samples} through the nonlinear function: {gab:2.2f}')
        print(f'  2) using the rules of error propagation through nonlinear functions: {y_function:2.2f}')
        print(f'  3) using basic rules (product) of error propagation: {y_prod:2.2f}')
        print()

        assert abs(gab.x - y_function.x) < 1e-10
        assert abs(gab.dx - y_function.dx) < 1e-1

    #----------------------------------------#
    def test_nonlinear_x3(self):
        """
        nonlinear operations (for the README.md):
        """

        # ground truth:
        val         = 10.0
        val_std     =  0.1
        # simulation: draw nb_samples with mean=val and std=val_std:
        nb_samples  = 10000
        val_samples = val + (val_std * np.random.randn(nb_samples))
        # nonlinear function:
        fn          = lambda x:x**3
        # alternative 1: samples through the nonlinear function:
        y_samples   = [fn(i) for i in val_samples]
        # alternative 2: using knumuncert.function() to propagate the nonlinearity:
        y_function  = knumuncert(val, val_std).function(fn)
        # alternative 3: direct product of two values with uncertainty:
        y_prod      = knumuncert(val, val_std) * knumuncert(val, val_std) * knumuncert(val, val_std)
        # printing the results:
        gab = knumuncert(fn(val), np.std(y_samples))
        print(f'\nFunction: y = x^3')
        print(f'  1) from {nb_samples} through the nonlinear function: {gab:2.2f}')
        print(f'  2) using the rules of error propagation through nonlinear functions: {y_function:2.2f}')
        print(f'  3) using basic rules (product) of error propagation: {y_prod:2.2f}')
        print()

        assert abs(gab.x - y_function.x) < 1e-10
        assert abs(gab.dx - y_function.dx) < 1e0

    #----------------------------------------#

#####################################################
