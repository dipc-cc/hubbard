import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.geometry as geom 
import numpy as np
import os

ch = __import__('chiral-geom')
op = __import__('open_boundary')

n = 3
m = [1,2]
w = [4,6,8]

for m_i in m:
    for w_i in w:
        geom = geom.cgnr(n,m_i,w_i).uc
        directory = '%i-%i-%i'%(n,m_i,w_i)
        print('Doing', directory)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        ch.analyze_edge(geom, directory)
        ch.analyze(geom, directory)
        ch.plot_states(geom, directory)
        ch.gap_exp(geom, directory)

        h = sisl.Hamiltonian(geom)
        for ia in geom:
            idx = geom.close(ia, R=[0., 1.43])
            h[ia, idx[0]] = 0
            h[ia, idx[1]] = -2.7
        # Plot surface and bulk density of states
        op.open_boundary(h, directory, xlim=0.5)

if False:
    # Plot bulk and surface DOS of 1D chain to test the funcion
    directory='./'
    print('Doing', directory)
    if not os.path.isdir(directory):
        os.mkdir(directory)

    geom = sisl.Geometry([[0,0,0]], atom='C', sc=[1.42,10,10])
    geom.set_nsc([3,1,1])
    h = sisl.Hamiltonian(geom)
    for ia in geom:
        idx = geom.close(ia, R=[0., 1.43])
        h[ia, idx[0]] = 0
        h[ia, idx[1]] = -2.7
    op.open_boundary(h, directory, xlim=7)