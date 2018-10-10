import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np
import sys

fn = sys.argv[1]
U = float(sys.argv[2])
eB = float(sys.argv[3])

H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=U, eB=eB, nsc=[3, 1, 1], kmesh=[51, 1, 1], what='xyz')
if U < 0.1:
    H.converge(premix=1.0)
else:
    H.converge()
H.save()

# Computation of band-resolved Zak-phases
nx = 100
zlist = []
for band in range(H.Nup+7):
    # compute Zak phase for bands near the gap
    zak = H.get_Zak_phase(Nx=nx, sub=band)
    zlist.append(zak)
    z2 = int(np.round(np.abs(1-np.exp(1j*zak))/2, 0))
    print 'Zak (Z2) for N=%i : %.4f (%i)'% (band, zak, z2)

# Spin polarization plot
p = plot.SpinPolarization(H, colorbar=True)
p.set_title(r'$e_\mathrm{B}=%.2f$ eV, $U=%.2f$ eV'%(eB, U))
fo = fn.replace('.XV', '-pol-U%.2f-eB%.2f.pdf'%(U, eB))
p.savefig('summary/'+fo)

# Bandstructure plot
ev = H.eigh(k=[0, 0, 0])
batoms = list(np.where(H.geom.atoms.Z == 5)[0])
p = plot.Bandstructure(H, scale=2, ymax=2, projection=batoms)
p.set_title(r'$e_\mathrm{B}=%.2f$ eV, $U=%.2f$ eV'%(eB, U))
for i, zak in enumerate(zlist):
    p.axes.annotate('%.2f'%zak, (0.13*(i%2), ev[i]-H.midgap), size=8)
fo = fn.replace('.XV', '-zak-U%.2f-eB%.2f.pdf'%(U, eB))
p.savefig('summary/'+fo)
