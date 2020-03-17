
Simulating an open-quantum system with electron correlations
============================================================

In this example we will create the `HubbardHamiltonian` object
of a system with open boundary conditions, and find the self-consistent solution using the mean field Hubbard Hamiltonian.

We will model a perfect system, where we *know* that we should recover a perfect step-like transmission function.
You can also navigate through the
`open-quantum systems section <https://github.com/dipc-cc/hubbard/tree/master/examples/open>`_ in the main repository,
where more examples can be found.

The sysem of this example is composed by a central region coupled to two electrodes (left and right). 
The central region in this case will be just a repetition of the unit-cell of the electrodes (perfect system),
for the sake of simplicity. We will consider the electrodes to be the unit cell of a zigzag graphene nanoribbon (ZGNR)
of width W=5 C-atoms across periodic along the x-axis.

We will focus on the equilibrium situation, therefore the temperature and chemical potentials of the electrodes *must coincide*.
The complex contour that we use to integrate the density matrix in the `Hubbard.NEGF` class is extracted from a `Transiesta <https://launchpad.net/siesta>`_ calculation
performed for a temperature of `kT=0.025` eV, which we will set as common for all the composing element calculations.

#. We will start by building the tight-binding (TB) Hamiltonian for the graphene nanoribbons,
   which compose the electrodes (periodic boundary conditions). Then we will find their self-consistent solution
   by using the `Hubbard` package.

#. We then have to build the geometry of the central region before we generate its sp2 TB Hamiltonian.

#. We will build the `HubbardHamiltonian` object, which will allow us to use the routines
   stored in this class to converge until we find the self-consistent solution.

#. In this case we will have to use a method based on the non-equilibrium Green's function of the central region 
   to obtain the spin-densities. To do so, we can make use of the methods available in the `Hubbard.NEGF` class.

#. Optionally, you can also compute the transmission function of the converged system using `TBtrans <https://launchpad.net/siesta>`_.

.. code-block:: python

      import Hubbard.hamiltonian as hh
      import Hubbard.geometry as geometry
      import Hubbard.sp2 as sp2
      import Hubbard.density as density
      from Hubbard.negf import NEGF

      # Build sisl.Geometry object of a ZGNR of width W=5 C-atoms across,
      # e.g., by using the function Hubbard.geometry.zgnr. 
      # This function returns a periodic ZGNR along the x-axis.
      ZGNR = geometry.zgnr(5)

      # Set U and kT for the whole calculation
      U = 2.0
      kT = 0.025

      # Build tight-binding Hamiltonian using sp2 function
      H_elec = sp2(ZGNR, t1=2.7, t2=0.2, t3=0.18, s1=0, s2=0, s3=0)
      # Hubbard Hamiltonian of elecs
      MFH_elec = hh.HubbardHamiltonian(H_elec, U=U, nkpt=[100, 1, 1], kT=kT)

      # Converge Electrode Hamiltonians
      dn = MFH_elec.converge(density.dm)

      # Central region is a repetition of the electrodes
      HC = H_elec.tile(3,axis=0)
      # without periodic boundary conditions
      HC.set_nsc([1,1,1])

      # Map electrodes in the device region, i.e., extract the atomic indices that correspond
      # to the electrodes inside the device region
      elec_indx = [range(len(H_elec)), range(len(HC.H)-len(H_elec), len(HC.H))]

      # MFH object of the central region
      MFH_HC = hh.HubbardHamiltonian(HC, U=U, kT=kT)
      # Use initial random density
      MFH_HC.random_density()

      # First create the NEGF object, where we pass the MFH converged electrodes and
      # the central region HubbardHamiltonian object
      negf = NEGF(MFH_HC, [MFH_elec, MFH_elec], elec_indx, elec_dir=['-A', '+A'])

      # Converge using Green's function method to obtain the densities
      dn = MFH_HC.converge(negf.dm_open, steps=1, tol=1e-6)

As you might probably observed, each iteration using the non-equilibrium Green's function formalism takes much more
time. This is due to the fact that it has to invert the Green's function matrix for each energy point along the
integration path in each iteration.
For this reason, using a good initial guess (initial spin-densities) can be crucial if we want the code to perform
less iterations to find the self-consistent solution.

Try again the example from above but using a specific initial guess for the spin-densities.
For instance, you could find the self-consistent solution for the device Hamiltonian with periodic boundary conditions
and use this spin-densities to start the convergence for the device with open boundary conditions.
