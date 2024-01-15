import numpy as np
import sisl


def real_space_grid(geometry, lattice, vector, shape, mode='wavefunction', **kwargs):
    """ Create real space `sisl.Grid` object for a `sisl.Geometry` and a `sisl.Supercell`

    Parameters
    ----------
    g : sisl.Geometry
        geometry
    vector : numpy.array
        vector to expand in the real space grid
    shape: float or (3,) of int
        the shape of the grid. A float specifies the grid spacing in Angstrom, while a list of integers specifies the exact grid size
        (see `sisl.Grid`)
    lattice : sisl.Lattice
        lattice object of the grid
    mode: str, optional
        to build the grid from sisl.electron.wavefunction object (``mode='wavefunction'``)
        or from the sisl.physics.DensityMatrix (``mode='charge'``), e.g. for charge-related plots

    See Also
    --------
    sisl.Grid

    """
    # Create a temporary copy of the geometry
    g = geometry.copy()

    # Set new lattice to create real-space grid
    g.set_lattice(lattice)

    # Create the real-space grid
    grid = sisl.Grid(shape, lattice=lattice, geometry=g)

    if mode in ['wavefunction']:
        if isinstance(vector, sisl.physics.electron.EigenstateElectron):
            # Set parent geometry equal to the temporary one
            vector.parent = g
            vector.wavefunction(grid)
        else:
            # In case v is a vector
            sisl.electron.wavefunction(vector, grid, geometry=g)
    elif mode in ['charge']:
        # The input vector v corresponds to charge-related quantities
        # including spin-polarization understood as charge_up - charge_dn
        D = sisl.physics.DensityMatrix(g)
        a = np.arange(len(D))
        D.D[a, a] = vector
        D.density(grid)

    del g
    return grid
