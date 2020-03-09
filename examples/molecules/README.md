
# Simulating a molecule with electron correlations #

In this example we will create the `HubbardHamiltonian` object
of a finite molecule, and find the self-consistent solution.

We will use as a reference the molecule of ref. [Nature Communications 10, 200 (2019)][Li2019] and compare the simulation with the experimental system.

   - We will start by building the tight-binding Hamiltonian for an sp2
   carbon-based molecule, by first reading the geometry file stored in [examples/molecules/kondo-paper/junction-2-2.XV][examples/junction]
   from the main repository, using [sisl][sisl]. You can find the parameters used here in the referenced paper above.

   - We will build the `HubbardHamiltonian` object, which will allow us to use the routines
   stored in this class to converge until we find the self-consistent solution.

   - We then can manipulate and obtain different magnetic states to compare the total energies
   that will tell us which one is the groundstate.

   - You can try to simulate more things looking at the [supp. material][Li2019:supp-material] of ref. [Nature Communications 10, 200 (2019)][Li2019].

    import sisl
    import Hubbard.hamiltonian as hh
    import Hubbard.sp2 as sp2
    import Hubbard.density as density
    import Hubbard.plot as plot

    # Build sisl.Geometry object from 'XV' file
    g = sisl.get_sile('molecules/kondo-paper/junction-2-2.XV').read_geometry()

    # Build tight-binding Hamiltonian using sp2 function
    Hsp2 = sp2(g, t1=2.7, t2=0.2, t3=0.18)

    # Build the HubbardHamiltonian object with U=3.5 at a temperature of kT=1e-5 in units
    # of the Boltzmann constant
    HH = hh.HubbardHamiltonian(Hsp2, U=3.5, kT=1e-5)

    # Let's initiate with a random density as a starting point
    HH.random_density()

    # Converge until a tolerance of tol=1e-10
    dn = HH.converge(density.dm, tol=1e-10)

    # Let's visualize some the final result:
    p = plot.SpinPolarization(HH, colorbar=True)
    p.savefig('spin.pdf')

<!---
Links to external and internal sites.
-->
[sisl]: https://sisl.readthedocs.io/en/latest/introduction.html
[examples/junction]: https://github.com/dipc-cc/hubbard/blob/master/examples/molecules/kondo-paper/junction-2-2.XV
[examples]: https://github.com/dipc-cc/hubbard/tree/master/examples
[Li2019]: https://www.nature.com/articles/s41467-018-08060-6
[Li2019:supp-material]: https://www.nature.com/articles/s41467-018-08060-6#Sec12
