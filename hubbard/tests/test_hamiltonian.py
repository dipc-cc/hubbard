import gc
import warnings

import numpy as np
import pytest

import hubbard.hamiltonian as hh
import hubbard.sp2 as sp2
import hubbard.density as density
import sisl


def test_quick():
    molecule = sisl.geom.agnr(7).tile(3, 0)
    molecule.set_nsc([1, 1, 1])
    Hsp2 = sp2(molecule)
    H = hh.HubbardHamiltonian(Hsp2, U=3.5)
    H.random_density()
    dn = H.iterate(density.calc_n_insulator, mixer=sisl.mixing.LinearMixer())
    H.write_density('test.nc')
    H.write_initspin('test.fdf')


def _initspin_hamiltonian():
    # zgnr(2): four carbon sites, one pz orbital each, with distinct polarizations
    H = hh.HubbardHamiltonian(sp2(sisl.geom.zgnr(2)), U=3.)
    H.n = np.array([[0.61, 0.32, 0.47, 0.55], [0.39, 0.68, 0.51, 0.41]])
    return H


def _read_initspin(fn):
    # parse the Spin.Total value and the (index, polarization) pairs of the DM.InitSpin block
    total, block, inside = None, [], False
    with open(fn) as f:
        for line in f:
            words = line.split()
            if line.startswith('Spin.Total'):
                total = float(words[1])
            elif line.startswith('%block DM.InitSpin'):
                inside = True
            elif line.startswith('%endblock DM.InitSpin'):
                inside = False
            elif inside:
                block.append((int(words[0]), float(words[1])))
    return total, block


def test_write_initspin_ext_geom_permuted(tmp_path):
    # ext_geom holds internal atoms 2, 3 and 0 in that order (atom 1 is missing),
    # plus an atom that is not in the Hamiltonian
    H = _initspin_hamiltonian()
    extra = sisl.Geometry([[30., 0., 0.]], atoms=sisl.Atom(6))
    ext = H.geometry.sub([2, 3, 0]).add(extra)
    fn = tmp_path / 'init.fdf'
    with pytest.warns(UserWarning, match='no match in ext_geom'):
        H.write_initspin(fn, ext_geom=ext)
    total, block = _read_initspin(fn)

    # reference: match each external atom to an internal one by its coordinates
    pol = H.n[0] - H.n[1]
    expected = {}
    for ia, xyz in enumerate(ext.xyz):
        d = np.linalg.norm(H.geometry.xyz - xyz, axis=1)
        if d.min() < 1e-6:
            expected[ia + 1] = pol[np.argmin(d)]
    assert sorted(i for i, _ in block) == sorted(expected) == [1, 2, 3]
    for i, p in block:
        assert p == pytest.approx(expected[i], abs=1e-6)
    # Spin.Total sums the polarizations of the internal atoms present in ext_geom only
    assert total == pytest.approx(pol[[2, 3, 0]].sum(), abs=1e-6)


def test_write_initspin_default_unchanged(tmp_path):
    # Stored reference: the text written by the code before the ext_geom fix (pins behavior)
    expected = (
        'Spin.Fix True\n'
        'Spin.Total -0.040000\n'
        '%block DM.InitSpin\n'
        '     1   0.220000 #  C    0.0000   10.0000   10.0000\n'
        '     2  -0.360000 #  C    0.0000   12.8400   10.0000\n'
        '     3  -0.040000 #  C    1.2298   10.7100   10.0000\n'
        '     4   0.140000 #  C    1.2298   12.1300   10.0000\n'
        '%endblock DM.InitSpin\n\n'
    )
    H = _initspin_hamiltonian()
    fn = tmp_path / 'init.fdf'
    H.write_initspin(fn)
    assert fn.read_text() == expected


def test_write_initspin_closes_file(tmp_path):
    H = _initspin_hamiltonian()
    fn = tmp_path / 'init.fdf'
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        H.write_initspin(fn)
        H.write_initspin(fn, ext_geom=H.geometry)
        gc.collect()
    assert not [w for w in record if issubclass(w.category, ResourceWarning)]


def test_write_initspin_ext_geom_no_match(tmp_path):
    # no atom in common: an empty block with Spin.Total 0 would pin SIESTA's moment to zero
    H = _initspin_hamiltonian()
    fn = tmp_path / 'init.fdf'
    with pytest.raises(ValueError, match='no atom of ext_geom'):
        H.write_initspin(fn, ext_geom=H.geometry.translate([50., 0., 0.]))
    assert not fn.exists()


def test_write_initspin_ext_geom_unmatched_warns(tmp_path):
    # atom 0 of ext_geom sits at a periodic image of internal atom 0, which overlap does not search
    H = _initspin_hamiltonian()
    ext = H.geometry.copy()
    ext.xyz[0] += ext.cell[0]
    fn = tmp_path / 'init.fdf'
    with pytest.warns(UserWarning, match='1 atoms of the Hamiltonian geometry have no match'):
        H.write_initspin(fn, ext_geom=ext)
    total, block = _read_initspin(fn)
    pol = H.n[0] - H.n[1]
    assert [i for i, _ in block] == [2, 3, 4]
    assert total == pytest.approx(pol[1:].sum(), abs=1e-6)


@pytest.mark.parametrize('case', ['two_external_near_one', 'eps_spans_neighbours'])
def test_write_initspin_ext_geom_ambiguous(tmp_path, case):
    H = _initspin_hamiltonian()
    if case == 'two_external_near_one':
        # a second external atom 0.05 Ang from internal atom 0
        near = sisl.Geometry([H.geometry.xyz[0] + [0.05, 0., 0.]], atoms=sisl.Atom(6))
        ext, eps = H.geometry.add(near), 0.1
    else:
        # eps larger than the 1.42 Ang bond: every atom matches its neighbours too
        ext, eps = H.geometry, 1.5
    fn = tmp_path / 'init.fdf'
    with pytest.raises(ValueError, match='more than one atom'):
        H.write_initspin(fn, ext_geom=ext, eps=eps)
    assert not fn.exists()
