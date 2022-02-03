import numpy as np
import sisl


def real_space_grid(geometry, v, grid_unit, xmin, xmax, ymin, ymax, z=1.1, mode='wavefunction', **kwargs):
    """
    Parameters
    ----------
    g : sisl.Geometry
    v : numpy.array
    grid_unit: 3-array-like
    mode: str, optinoal
        to build the grid from sisl.electron.wavefunction object (``mode=wavefunction``)
        or from the sisl.physics.DensityMatrix (``mode=`charge`), e.g. for charge-related plots
    sc: sisl.SuperCell, optional
        super cell, defaults to the sc saved in the geometry
    """
    # Create a temporary copy of the geometry
    g = geometry.copy()

    # Set new sc to create real-space grid
    sc = sisl.SuperCell([xmax-xmin, ymax-ymin, 1000], origin=[0, 0, -z])
    g.set_sc(sc)

    # Move geometry within the supercell
    g = g.move([-xmin, -ymin, -np.amin(g.xyz[:, 2])])
    # Make z~0 -> z = 0
    g.xyz[np.where(np.abs(g.xyz[:, 2]) < 1e-3), 2] = 0

    # Create the real-space grid
    grid = sisl.Grid(grid_unit, sc=sc, geometry=g)

    if mode in ['wavefunction']:
        if isinstance(v, sisl.physics.electron.EigenstateElectron):
            # Set parent geometry equal to the temporary one
            v.parent = g
            v.wavefunction(grid)
        else:
            # In case v is a vector
            sisl.electron.wavefunction(v, grid, geometry=g)
    elif mode in ['charge']:
        # The input vector v corresponds to charge-related quantities
        # including spin-polarization understood as charge_up - charge_dn
        D = sisl.physics.DensityMatrix(g)
        a = np.arange(len(D))
        D.D[a, a] = v
        D.density(grid)
    
    if 'smooth' in kwargs:
        # Smooth grid with gaussian function
        if 'r_smooth' in kwargs:
            r_smooth = kwargs['r_smooth']
        else:
            r_smooth = 0.7
        grid = grid.smooth(method='gaussian', r=r_smooth)

    # Slice it to obtain a 2D grid
    grid = grid.grid[:, :, 0].T.real

    del g
    return grid
