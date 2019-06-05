import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.ncdf as ncdf
import numpy as np
import sisl

# Test all plot functionalities of Hubbard module
# using a reference molecule (already converged)

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.sc.set_nsc([1,1,1])
molecule = molecule.move(-molecule.center(what='xyz')).rotate(220, [0,0,1])

calc = ncdf.read('mol-ref/mol-ref.nc')
H = hh.HubbardHamiltonian(molecule)
H.U = calc.U
H.Nup, H.Ndn = calc.Nup, calc.Ndn
H.nup, H.ndn = calc.nup, calc.ndn
H.update_hamiltonian()

p = plot.Charge(H, colorbar=True)
p.savefig('chg.pdf')

p = plot.Charge(H, realspace=True, colorbar=True)
p.savefig('chg_rs.pdf')

p = plot.ChargeDifference(H)
p.savefig('chgdiff.pdf')

p = plot.ChargeDifference(H, realspace=True)
p.savefig('chgdiff_rs.pdf')

p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol.pdf')

p = plot.SpinPolarization(H, realspace=True)
p.savefig('pol_rs.pdf')

ev, evec = H.eigh(eigvals_only=False, spin=0)
p = plot.Wavefunction(H, 500*evec[:, 10])
p.savefig('wf.pdf')

p = plot.Wavefunction(H, 10*evec[:, 10], realspace=True)
p.savefig('wf_rs.pdf')

p = plot.Spectrum(H)
p.savefig('spectrum.pdf')

p = plot.LDOSmap(H)
p.savefig('ldos.pdf')

p = plot.DOS_distribution(H, 0.15, f=300, sites=[60])
p.savefig('dos.pdf')

p = plot.DOS(H, np.linspace(-0.2,0.2,101))
p.savefig('dos2.pdf')

p = plot.DOS(H, np.linspace(-0.2,0.2,101), sites=[60])
p.savefig('ldos2.pdf')
