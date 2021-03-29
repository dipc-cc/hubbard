
Simulating a molecule with electron correlations
================================================

In this example we will create the `HubbardHamiltonian` object
of a finite molecule, and find the self-consistent solution.

We will use as a reference the molecule of ref. `Nature Communications 10, 200 (2019) <https://www.nature.com/articles/s41467-018-08060-6>`_
and compare the simulation with the experimental system.

You can also navigate through the 
`molecule examples section <https://github.com/dipc-cc/hubbard/tree/master/examples/molecules>`_,
where more molecular geometries suitable to compute with the mean field hubbard model can be found.

#. We will start by building the tight-binding (TB) Hamiltonian for an sp2 
   carbon-based molecule, by first reading the geometry file stored in this `file <https://github.com/dipc-cc/hubbard/blob/master/examples/molecules/kondo-paper/junction-2-2.XV>`_
   (which you can download) using `sisl <https://sisl.readthedocs.io/en/latest/index.html>`_.
   You can find the parameters used to model this sp2 TB Hamiltonian
   in the `Supp. Material <https://www.nature.com/articles/s41467-018-08060-6#Sec12>`_ of the paper referenced above.

#. We will build the `HubbardHamiltonian` object, which will allow us to use the routines
   stored in this class to converge the mean field hubbard Hamiltonian until we find the self-consistent solution.

#. We then can manipulate and obtain different magnetic states to compare the total energies
   that will tell us which one is the groundstate.

#. You can try to reproduce more calculations by looking at the `Supp. Material <https://www.nature.com/articles/s41467-018-08060-6#Sec12>`_
   of the above referenced paper.

.. code-block:: python

      import sisl
      import hubbard.hamiltonian as hh
      import hubbard.sp2 as sp2
      import hubbard.density as density
      import hubbard.plot as plot

      # Build sisl.Geometry object from the 'XV' file
      g = sisl.get_sile('junction-2-2.XV').read_geometry()

      # Build sisl.Hamiltonian object using the sp2 function
      Hsp2 = sp2(g, t1=2.7, t2=0.2, t3=0.18)

      # Build the HubbardHamiltonian object with U=3.5 at a temperature of kT=1e-5 
      # in units of the Boltzmann constant
      HH = hh.HubbardHamiltonian(Hsp2, U=3.5, kT=1e-5)

      # Let's initiate with a random density as a starting point
      HH.random_density()

      # Converge until a tolerance of tol=1e-10
      dn = HH.converge(density.dm, tol=1e-10)

      # Save total energy
      E_0 =  HH.Etot

      # Let's visualize the final result:
      p = plot.SpinPolarization(HH, colorbar=True, vmax=0.4, vmin=-0.4)
      p.savefig('spin.pdf')

      # Until now, in absence of specifications we have found the antiferromagnetic solution
      # (same up- and down-spin particles). But we can compute the solution for the
      # ferromagnetic case, by imposing Qup = Qdn + 2 and converge again

      HH.q[0] += 1
      HH.q[1] -= 1

      # Converge until a tolerance of tol=1e-10
      dn = HH.converge(density.dm, tol=1e-10)

      # Let's visualize some the final result:
      p = plot.SpinPolarization(HH, colorbar=True, vmax=0.4, vmin=-0.4)
      p.savefig('spin-ferro.pdf')

      # Compare energies between the two calculations
      print('E_FM - E_AFM : ', HH.Etot - E_0, ' eV')

At this point we found that the groundstate of this molecule is the antiferromagnetic solution, since it is lower
in energy compared to the ferromagnetic state.

Furthermore, we can predict the magnetic groundstate in bipartite lattices by using the `Lieb's theorem <https://link.aps.org/doi/10.1103/PhysRevLett.62.1201>`_.
This theorem states that the total spin (:math:`S`) of the groundstate for a bipartite lattice is proportional to the
imbalance between the A and B sublattices:

.. math::
   S = \frac{1}{2}|N_{A}-N_{B}|

Therefore, if we create an imbalance between the A and B sublattices,
we will find that the state with lower energy appears for :math:`S>0`.

A real example to this can be found in ref. `arXiv:1912.08298 <https://arxiv.org/abs/1912.08298>`_. Where triangular shaped
graphene nanoflakes show a groundstate with :math:`S=1`.
