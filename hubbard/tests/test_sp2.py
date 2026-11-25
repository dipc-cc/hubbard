import numpy as np
import pytest
import sisl

from hubbard import sp2


@pytest.mark.parametrize("parameter", ["s2", "s3"])
def test_sp2_overlap_beyond_first_neighbor_needs_s1(parameter):
    with pytest.raises(ValueError, match="nonzero first-neighbor overlap"):
        sp2(sisl.geom.graphene(), **{parameter: 0.1})


def test_sp2_default_orthogonal():
    assert sp2(sisl.geom.graphene()).orthogonal


def test_sp2_overlap_shells_with_s1():
    # Reference from the geometry alone: 1 on site and s1, s2, s3 on the
    # first, second and third neighbor shells (1.42, 2.46, 2.84 Ang)
    g = sisl.geom.graphene()
    H = sp2(g, s1=0.1, s2=0.01, s3=0.05)
    assert not H.orthogonal
    r = g.rij(np.arange(g.na)[:, None], np.arange(g.na_s)[None, :])
    expected = np.select([r < 0.1, r < 1.6, r < 2.6, r < 3.1], [1.0, 0.1, 0.01, 0.05], 0.0)
    assert np.allclose(H.tocsr(H.S_idx).toarray(), expected, rtol=0, atol=1e-12)
