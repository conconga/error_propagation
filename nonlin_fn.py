from cnumincert import CNUMINCERT
import numpy            as np
import matplotlib.pylab as plt

if __name__ == "__main__":
    v       = 10
    sigma_v = 0.1
    nb_smps = 10000

    v_smp   = v + (sigma_v * np.random.randn(nb_smps))

    if 2 == 2:
        fn      = lambda x:x**2
        v_fn    = [fn(i) for i in v_smp]
        v2_fn1  = CNUMINCERT(v, sigma_v).function(fn)
        v2_fn2  = CNUMINCERT(v, sigma_v) * CNUMINCERT(v, sigma_v)

    else:
        fn      = lambda x:x**3
        v_fn    = [fn(i) for i in v_smp]
        v2_fn1  = CNUMINCERT(v, sigma_v).function(fn)
        v2_fn2  = CNUMINCERT(v, sigma_v) * CNUMINCERT(v, sigma_v) * CNUMINCERT(v, sigma_v)


    print("from samples, fn(v)  = {:2.2f} +- {:2.2f}".format(fn(v), np.std(v_fn)))
    print("from er_prop, v2_fn1 = {:2.2f} +- {:2.2f}".format(v2_fn1.x, v2_fn1.dx))
    print("from er_prop, v2_fn2 = {:2.2f} +- {:2.2f}".format(v2_fn2.x, v2_fn2.dx))

