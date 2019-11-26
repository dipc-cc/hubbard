import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import Hubbard.geometry as geometry 
import numpy as np
import os

ch = __import__('chiral-geom')
op = __import__('open_boundary')

# Choose model (1NN, 3NN):
model = '1NN'
if model == '3NN':
    t2, t3 = 0.2, 0.18
else:
    t2 = t3 = 0

# Compute and plot the bandgap as a function of (n,m) for a paritcular ribbon width
if False:
    m = [1,2,3,4,5]
    ch.plot_band_gap_imshow(m=m, figsize=(8,4), model=model, lim=1.0)

n = 3
m = [1,2]
w = [4,6,8]

for m_i in m:
    for w_i in w:
        geom = geometry.cgnr(n,m_i,w_i)
        geom.write('test.xyz')
        directory = '%i-%i-%i'%(n,m_i,w_i)
        print('Doing', directory)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        # TB model
        H0 = sp2(geom, t1=2.7, t2=t2, t3=t3)

        ch.analyze(H0, directory, model=model)
        # Make finite ribbon of 15 reps
        Hfinite = H0.tile(15, axis=0)
        Hfinite.set_nsc([1,1,1])
        ch.analyze_edge(Hfinite, directory, model=model)
        ch.plot_states(H0, directory, model=model)
        ch.gap_exp(H0, directory, model=model)

        # Plot surface and bulk density of states
        if m_i == 1 and w_i == 8:
            xlim=0.1
        else:
            xlim=0.5
        op.open_boundary(H0, directory, xlim=xlim, model=model)
