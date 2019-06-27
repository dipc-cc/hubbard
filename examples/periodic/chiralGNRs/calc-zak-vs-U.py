import sisl
import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2
import Hubbard.plot as plot
import Hubbard.geometry as geometry
import numpy as np
import os

model = '1NN'
#model = '3NN'
n, m, w = 3, 1, 4

s = '%i-%i-%i' % (n, m, w)

geom = geometry.cgnr(n, m, w)

if '3NN' in model:
    H0 = sp2(geom, t1=2.7, t2=0.2, t3=0.18)
elif '1NN' in model:
    H0 = sp2(geom, t1=2.7, t2=0., t3=0.)

U = np.linspace(5, 0, 21)
Z = 0*U

nx = 101

H = hh.HubbardHamiltonian(H0, U=max(U), nkpt=[nx, 1, 1])
success = H.read_density(s+'.nc')
if not success:
    H.polarize_sublattices()

for i, u in enumerate(U):
    # Iterate
    H.U = u
    H.read_density(s+'.nc')
    H.iterate(mix=0.1)
    H.converge(mix=0.95, steps=20, fn=s+'.nc')
    H.write_density(s+'.nc')

    # Zak
    zak = H.get_Zak_phase(Nx=nx)
    Z[i] = zak
    z2 = int(round(np.abs(1-np.exp(1j*zak))/2))
    print('U=%.2feV: Zak=%.3f' %(u, zak) )

    # Postprocess
    p = plot.SpinPolarization(H, colorbar=True, vmax=0.5)
    p.annotate()
    p.set_title('$U=%.2f$ eV' % H.U)
    p.savefig(s+'/%s-pol_%.2f.pdf' % (model, H.U) )
    p.close()

    ymax = 8.0
    p = plot.Bandstructure(H, ymax=ymax)
    #p.axes.annotate(r'$\mathbf{Z_2=%i}$' % z2, (0., 0.9*ymax), size=22, backgroundcolor='k', color='w')
    p.axes.annotate(r'$\phi/\pi=%.2f$' % (zak/np.pi), (0.5, 0.9*ymax), size=16)
    p.savefig(s+'/bands_1NN_U%.2f.pdf' % H.U)
    p.close()

# Translate phases by 2pi:
idx = np.where(Z > 0.5)[0]
Z[idx] -= 2*np.pi

p = plot.Plot(figsize=(6, 4))
p.axes.plot(U, Z/np.pi, marker='o',)
p.set_xlabel('$U$ (eV)')
p.set_ylabel('Zak phase $\phi/\pi$')
p.set_title('(%i, %i, %i)-cGNR' % (n, m, w) )
p.savefig(s+'/%s-phase-vs-U.pdf' % model)
