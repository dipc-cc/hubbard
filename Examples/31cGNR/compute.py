import Hubbard.hubbard as HH
import sys
import numpy as np

# USE:
# python compute.py <geometry.XV>

# Density tolerance for quitting the iterations
tol = 1e-7

# 3NN tight-binding model
H = HH.Hubbard(sys.argv[1], t1=2.7, t2=0.2, t3=.18, nsc=[3, 1, 1], kmesh=[51, 1, 1])

for u in np.linspace(0.0,4.0,5):
    print
    print 'U =', u
    H.U = u
    H.read()
    dN, E = H.iterate(mix=1.0)
    while dN > 1e-2 and u > 0:
        dN, E = H.iterate(mix=.2)
        print dN
    i = 0
    while dN > tol and u > 0:
        dN, E = H.iterate(mix=1.)
        print dN
        i += 1
        if i%100 == 0:
            H.save()
    H.save()
    egap, emid = H.find_midgap()
    print u, egap, 'tmp'
    H.plot_bands()
    H.plot_polarization()
    if u == 0:
        # Fix some issues with the first plot
        H.plot_bands()
