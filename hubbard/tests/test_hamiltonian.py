import pytest

import hubbard.hamiltonian as hh
import hubbard.sp2 as sp2
import hubbard.density as dm
import sisl


def test_quick():
    molecule = sisl.geom.agnr(7).tile(3, 0)
    molecule.set_nsc([1, 1, 1])
    Hsp2 = sp2(molecule)
    H = hh.HubbardHamiltonian(Hsp2, U=3.5)
    H.random_density()
    dn = H.iterate(dm.dm_insulator, mixer=sisl.mixing.LinearMixer())
    H.write_density('test.nc')
    H.write_initspin('test.fdf')
