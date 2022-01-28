from hubbard import HubbardHamiltonian, sp2, density, plot
import numpy as np
import sisl

# Test all plot functionalities of hubbard module
# using a reference molecule (already converged)

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.sc.set_nsc([1, 1, 1])
molecule = molecule.move(-molecule.center(what='xyz')).rotate(220, [0, 0, 1])
H_mol = sp2(molecule)

p = plot.BondHoppings(H_mol, annotate=False, off_diagonal_only=False, cmap_e='winter')
p.legend()
p.savefig('bondHoppings.pdf')

H = HubbardHamiltonian(H_mol, U=3.5)
H.read_density('mol-ref/density.nc')
H.iterate(density.calc_n_insulator)

p = plot.Charge(H, ext_geom=molecule, colorbar=True)
p.savefig('chg.pdf')

p = plot.ChargeDifference(H, ext_geom=molecule, colorbar=True)
p.savefig('chgdiff.pdf')

p = plot.SpinPolarization(H, ext_geom=molecule, colorbar=True, vmax=0.2)
p.annotate()
p.savefig('pol.pdf')

H.H.shift(-H.find_midgap())
ev, evec = H.eigh(eigvals_only=False, spin=0)
p = plot.Wavefunction(H, 500 * evec[:, 10], ext_geom=molecule, colorbar=True)
p.savefig('wf.pdf')

p = plot.Spectrum(H)
p.savefig('spectrum.pdf')

p = plot.LDOSmap(H)
p.savefig('ldos_map.pdf')

p = plot.DOS(H, np.linspace(-0.2, 0.2, 101))
p.savefig('total_dos.pdf')

p = plot.DOS(H, np.linspace(-0.2, 0.2, 101), sites=[60])
p.savefig('ldos.pdf')

DOS = H.PDOS(ev[10], eta=1e-3, spin=[0])
p = plot.DOS_distribution(H, DOS * 100, sites=[60], ext_geom=molecule)
p.savefig('pdos.pdf')

# Test real-space plots?
if True:

    p = plot.DOS_distribution(H, evec[:, 10], z=5, vmax=2e-15, realspace=True, ext_geom=molecule, colorbar=True)
    p.savefig('pdos_rs.pdf')

    p = plot.Charge(H, realspace=True, ext_geom=molecule, vmax=1e-4, vmin=-1e-4, colorbar=True)
    p.savefig('chg_rs.pdf')

    p = plot.ChargeDifference(H, ext_geom=molecule, realspace=True, vmax=3e-5, vmin=-3e-5, colorbar=True)
    p.savefig('chgdiff_rs.pdf')

    p = plot.SpinPolarization(H, ext_geom=molecule, realspace=True, vmax=3e-5, vmin=-3e-5, colorbar=True)
    p.savefig('pol_rs.pdf')

    p = plot.Wavefunction(H, evec[:, 10], ext_geom=molecule, realspace=True, vmax=1.5e-3, vmin=-1.5e-3, colorbar=True)
    p.savefig('wf_rs.pdf')
