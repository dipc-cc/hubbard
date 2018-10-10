import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np
import sys

fn = sys.argv[1]
U = float(sys.argv[2])

H = hh.HubbardHamiltonian(fn, t1=2.7, t2=0.2, t3=.18, U=U, nsc=[3, 1, 1], kmesh=[51, 1, 1], what='xyz')
H.converge()
H.save()

p = plot.SpinPolarization(H, colorbar=True)
p.annotate()
p.savefig('polarization.pdf')

# Project onto boron sites
batoms = list(np.where(H.geom.atoms.Z == 5)[0])
p = plot.Bandstructure(H, scale=2, projection=batoms)
p.set_title(r'$U=%.2f$ eV: %s'%(U, fn.replace('_', '-')))
p.savefig('bands_U%.2f.pdf'%U)
