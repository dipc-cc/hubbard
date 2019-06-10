from __future__ import print_function
import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt

fn = sys.argv[1]

# Read geometry and set up SSH Hamiltonian
geom = sisl.get_sile(fn).read_geometry()
geom = geom.tile(2,axis=0)
geom = geom.move(-geom.center(what='xyz'))
geom.set_nsc([3, 1, 1])
H = sisl.Hamiltonian(geom)
for ia in geom:
    idx = geom.close(ia, R=[0.1, 1.1, 2.1])
    H[ia, idx[0]] = 0.
    H[ia, idx[1]] = 1.0 # 1NN
    H[ia, idx[2]] = 0.5 # 2NN

zak_Cl = []
zak_Op = []
Nx = np.arange(10, 151)
for nx in Nx:

    def func(sc, frac):
        return [-0.5+frac+0.5/nx, 0, 0]
    # Closed loop, show that this leads to incorrect results
    bzCl = sisl.BrillouinZone(H).parametrize(H, func, nx)
    print(bzCl.k)


    def func2(sc, frac):
        return [-0.5+1.*nx/(nx-1)*frac, 0, 0]
    # Open loop, correct integration contour for Zak phase
    bzOp = sisl.BrillouinZone(H).parametrize(H, func2, nx)
    #print(bzOp.k)

    for band in [[range(len(H)/2)]]:
        print('\nBand index =', band)
        zak = sisl.electron.berry_phase(bzCl, sub=band, closed=True, method='zak')
        zak_Cl.append(zak)
        print('Zak (closed) : %.4f rad' % zak)
        zak = sisl.electron.berry_phase(bzOp, sub=band, closed=False)
        zak_Op.append(zak)
        print('Zak (open)   : %.4f rad' % zak)
        z2 = int(np.abs(1-np.exp(1j*zak))/2)
        print('Z2 invariant =', z2)

import Hubbard.plot as plot
p = plot.Plot()
p.axes.plot(Nx, np.array(zak_Cl)/np.pi, 'ro', label=r'Zak Closed')
p.axes.plot(Nx, np.array(zak_Op)/np.pi, 'ko', label=r'Zak Open')
p.set_ylabel(r'Zak Phase [$\pi$]')
p.set_xlabel(r'Number of $k$-points ($n_{x}$)')
p.set_ylim(-1.5,1.5)
p.axes.legend()
p.set_title(fn[:-3]+' cell x 2')
p.savefig(fn[:-3]+'_zak_nx-2cells.pdf')

