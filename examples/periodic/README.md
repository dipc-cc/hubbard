# Periodic systems #

In this example we will create the `HubbardHamiltonian` object
of a system with periodic boundary conditions, and find the self-consistent solution using the mean field Hubbard model.

We will use as a reference the graphene nanoribbons of ref. [Phys. Rev. B 81, 245402 (2010)][Hancock2010] and benchmark the results published there.

   - We will start by building the tight-binding (TB) Hamiltonian for the sp2 carbon-based
    graphene nanoribbon, by first building the [sisl.Geometry][sisl-geom] object. 
    You can find the parameters used to model the TB Hamiltonian here in the referenced paper above.

   - We will build the `HubbardHamiltonian` object, which will allow us to use the routines stored in this class to converge until we find the self-consistent solution.

   - Then we can plot some physical quantities that may be relevant after the calculation with the Hubbard package.

    import Hubbard.hamiltonian as hh
    import Hubbard.geometry as geometry
    import Hubbard.sp2 as sp2
    import Hubbard.density as density
    import Hubbard.plot as plot

    # Build sisl.Geometry object of a zigzag graphene nanoribbon of width W=16 C-atoms
    # across, e.g., by using the function geometry.zgnr. This function returns a 
    # periodic graphene ribbon along the x-axis.
    g = geometry.zgnr(16)

    # Build tight-binding Hamiltonian using sp2 function
    Hsp2 = sp2(g, t1=2.7, t2=0.2, t3=0.18, s1=0, s2=0, s3=0)

    # Build the HubbardHamiltonian object with U=2.0 eV at a temperature of kT=1e-5 in units 
    # of the Boltzmann constant.
    # We have to use 
    HH = hh.HubbardHamiltonian(Hsp2, U=2.0, nkpt=[100, 1, 1], kT=1e-5)

    # Let's plot the band structure of the pure tight-binding Hamiltonian
    # i.e., before finding the mean-field self-consistent solution
    p = plot.Bandstrucure(HH)
    p.savefig('bands_TB.pdf')

    # Let's initiate with a random density as a starting point
    HH.random_density()

    # Converge until a tolerance of tol=1e-10
    dn = HH.converge(density.dm, tol=1e-10)

    # Let's plot some relevant physical quantities, such as the final spin-polarization per unit-cell
    # and the band structure of the self-consistent solution to compare with the non-correlated solution
    p = plot.Bandstrucure(HH)
    p.savefig('bands_MFH.pdf')



<!---
Links to external and internal sites.
-->
[sisl-geom]: https://sisl.readthedocs.io/en/latest/api-generated/sisl.Geometry.html
[examples]: https://github.com/dipc-cc/hubbard/tree/master/examples
[Hancock2010]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.245402
