import Hubbard.hamiltonian as hh
import Hubbard.density as dens
import Hubbard.sp2 as sp2
import numpy as np
import sisl

# Test quantitatively that densities and eigenvalue spectrum are unchanged
# using a reference molecule (already converged)

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.sc.set_nsc([1, 1, 1])

# Try reading from file
Hsp2 = sp2(molecule)
H = hh.HubbardHamiltonian(Hsp2, U=3.5)
H.read_density('mol-ref/density.nc')
H.iterate(dens.dm_insulator, mixer=sisl.mixing.LinearMixer())

# Determine reference values for the tests
ev0, evec0 = H.eigh(eigvals_only=False, spin=0)
Etot0 = H.Etot*1

mixer = sisl.mixing.PulayMixer(0.7, history=7)

for m in [dens.dm_insulator, dens.dm]:
    # Reset density and iterate
    H.random_density()
    mixer.clear()
    dn = H.converge(m, tol=1e-10, steps=10, mixer=mixer, print_info=True)
    ev1, evec1 = H.eigh(eigvals_only=False, spin=0)

    # Total energy check:
    print('Total energy difference: %.4e eV' %(Etot0-H.Etot))

    # Eigenvalues are easy to check
    if np.allclose(ev1, ev0):
        print('Eigenvalue check passed')
    else:
        # Could be that up and down spins are interchanged
        print('Warning: Engenvalues for up-spins different. Checking down-spins instead')
        ev1, evec1 = H.eigh(eigvals_only=False, spin=1)
        if np.allclose(ev1, ev0):
            print('Eigenvalue check passed')

    # Eigenvectors are a little more tricky due to arbitrary sign
    if np.allclose(np.abs(evec1), np.abs(evec0)):
        print('Eigenvector check passed\n')
    else:
        print('Eigenvector check failed!!!\n')

if True:
    # Test also Hubbard.negf in the WBL approximation for gamma=0.
    # We will compare with the density and Etot obtained from diagonalization
    # at kT=0.025 (temperature of the CC)
    H.kT = 0.025
    H.random_density()
    mixer.clear()
    dn = H.converge(dens.dm, tol=1e-10, steps=10, mixer=mixer)
    Etot0 = H.Etot
    dm = 1*H.dm

    # Obtain DOS for the finite molecule with Lorentzian distribution
    egrid = np.linspace(-1,1,50)
    import Hubbard.plot as plot
    p = plot.DOS(H, egrid, eta=1e-2, spin=[0])

    # Now compute same molecule with the WBL approximation with gamma=0
    from Hubbard.negf import NEGF
    negf = NEGF(H, [],[], WBL=True, gamma=[0.], gamma_indx=[[0, 1]])
    # This is to use a better guess for the device potential
    H.find_midgap()
    negf.Ef = -H.midgap
    negf.eta = 1e-2
    mixer.clear()
    ddm = H.converge(negf.dm_open, mixer=mixer, tol=1e-10, func_args={'qtol':1e-4}, steps=10, print_info=True)
    print('Total energy difference: %.4e eV' %(Etot0-H.Etot))
    print('Density difference (up, dn): (%.4e, %.4e)'%(max(abs(H.dm[0]-dm[0])), max(abs(H.dm[1]-dm[1]))))

    # Plot DOS calculated from the diagonalization and the WBL with gamma=0
    dos = negf.DOS(H, egrid-negf.Ef, spin=0)
    p.axes.plot(egrid, dos, '--', label='WBL')
    p.legend()
    p.savefig('DOS-comparison.pdf')
