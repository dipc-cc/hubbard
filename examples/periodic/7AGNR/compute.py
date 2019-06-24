import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.ncdf as ncdf
import numpy as np
import sys
import sisl

fn = sys.argv[1]
geom = sisl.get_sile(fn).read_geometry()
geom = geom.move(-geom.center(what='xyz'))
geom.set_nsc([3,1,1])

U = float(sys.argv[2])
eB = float(sys.argv[3])

H = hh.HubbardHamiltonian(geom, t1=2.7, t2=0.2, t3=.18, U=U, eB=eB, kmesh=[51, 1, 1])
try:
    c = ncdf.read(fn[:-3]+'.nc') # Try reading, if we already have density on file
    H.nup, H.ndn = c.nup, c.ndn
except:
    H.random_density()

if U < 0.1:
    H.converge(premix=1.0)
else:
    H.converge()

ncdf.write(H, fn[:-3]+'.nc')

# Spin polarization plot
p = plot.SpinPolarization(H, colorbar=True, vmax=0.1)
p.set_title(r'$\varepsilon_\mathrm{B}=%.2f$ eV, $U=%.2f$ eV'%(eB, U))
fo = fn.replace('.XV', '-pol-U%.2f-eB%.2f.pdf'%(U, eB))
p.savefig('summary/'+fo)

# Bandstructure plot
ymax = 2
ev = H.eigh(k=[0, 0, 0])
batoms = list(np.where(H.geom.atoms.Z == 5)[0])
p = plot.Bandstructure(H, scale=2, ymax=ymax, projection=batoms)
p.set_title(r'$\varepsilon_\mathrm{B}=%.2f$ eV, $U=%.2f$ eV'%(eB, U))

# Sum over filled bands:
zak = H.get_Zak_phase(Nx=100)
z2 = int(round(np.abs(1-np.exp(1j*zak))/2))
tol = 0.05
if np.abs(zak) < tol or np.abs(np.abs(zak)-np.pi) < tol:
    # Only append Z2 when appropriate:
    p.axes.annotate(r'$\mathbf{Z_2=%i}$'%z2, (0., 0.9*ymax), size=22, backgroundcolor='k', color='w')
fo = fn.replace('.XV', '-zak-U%.2f-eB%.2f.pdf'%(U, eB))
p.savefig('summary/'+fo)
