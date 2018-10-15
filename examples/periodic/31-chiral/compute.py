import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np
import sys
import os

U = float(sys.argv[1])


def compute(fn):
    head, tail = os.path.split(fn)
    H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=U, nsc=[3, 1, 1], kmesh=[51, 1, 1], what='xyz')
    dn, etot = H.iterate()
    if dn > 0.1:
        # We don't have a good solution, try polarizing one edge:
        H.set_polarization([1, 6])
    if U == 0:
        H.converge(save=True, premix=1.0)
    else:
        H.converge(save=True, steps=25)
    H.save()

    p = plot.SpinPolarization(H, colorbar=True, figsize=(6, 10), vmax=0.2)
    #p.annotate()
    p.set_title(r'$U=%.2f$ eV [%s]'%(U, head))
    p.savefig(head+'/pol_U%.3i.pdf'%(U*100))

    ymax = 4.0
    p = plot.Bandstructure(H, ymax=ymax)
    p.set_title(r'$U=%.2f$ eV [%s]'%(U, head))
    # Sum over filled bands:
    zak = H.get_Zak_phase(Nx=100)
    p.axes.annotate(r'$\gamma=%.4f$'%zak, (0.4, 0.50), size=22, backgroundcolor='w')
    z2 = int(round(np.abs(1-np.exp(1j*zak))/2))
    tol = 0.05
    if np.abs(zak) < tol or np.abs(np.abs(zak)-np.pi) < tol:
        # Only append Z2 when appropriate:
        p.axes.annotate(r'$\mathbf{Z_2=%i}$'%z2, (0., 0.9*ymax), size=22, backgroundcolor='k', color='w')
    p.savefig(head+'/bands_U%.3i.pdf'%(U*100))

for fn in sys.argv[2:]:
    compute(fn)
