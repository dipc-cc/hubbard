
Simulating an open-quantum system with electron correlations
============================================================

In this example we will create the `HubbardHamiltonian` object
of a system with open boundary conditions, and find the self-consistent solution using the mean field Hubbard model.

We will model a perfect system, where we *know* that we must recover a perfect step-like transmission function.
You can also to navigate through the 
`examples/periodic <https://github.com/dipc-cc/hubbard/tree/master/examples/periodic>`_ section, where more examples can be found.

The sysem of this example is composed by a central region coupled to two electrodes (left and right). 
We have to use the non-equilibrium Green's function formalism to solve the spin-densities. The central region in this case will
be just a repetition 

We will focus in the equilibrium situation, therefore the temperature and chemical potentials of the electrodes *must coincide*.
The complex contour that we use to integrate the density matrix in the Hubbard.NEGF class is extracted from a Transiesta calculation
for a temperature of `kT=0.025` eV, which we will have to set as common for all the composing element calculations.

#. We will start by building the tight-binding (TB) Hamiltonian for the graphene nanoribbons,
   which compose the electrodes (periodic boundary conditions). Then we will find the self-consistent solution for them.
   You can find the parameters used here in the ref. paper above.

#. We then have to build the geometry of the central region before we generate its sp2 TB Hamiltonian.

#. We will build the `HubbardHamiltonian` object, which will allow us to use the routines
   stored in this class to converge until we find the self-consistent solution.
   In this case we will have to use a method to obtain the spin-densities based on the non-equilibrium
   Green's function of the central region, which contains the semi-infinite leads (electrodes).

.. code-block:: python

    import Hubbard.hamiltonian as hh
    import Hubbard.geometry as geometry
    import Hubbard.sp2 as sp2
    import Hubbard.negf as negf

    # Build sisl.Geometry object of a zigzag graphene nanoribbon of width W=5 C-atoms across, e.g., by using the function
    # geometry.zgnr. This function returns a periodic graphene ribbon along the x-axis.
    ZGNR = geometry.zgnr(5)

    # and 3NN TB Hamiltonian
    H_elec = sp2(ZGNR, t1=2.7, t2=0.2, t3=0.18)

    # Set U and kT for the whole calculation
    U = 2.0
    kT = 0.025

    # Build tight-binding Hamiltonian using sp2 function
    Hsp2 = sp2(g, t1=2.7, t2=0.2, t3=0.18, s1=0, s2=0, s3=0)
    # Hubbard Hamiltonian of elecs
    MFH_elec = hh.HubbardHamiltonian(H_elec, U=U, nkpt=[100, 1, 1], kT=kT)

    # Converge Electrode Hamiltonians
    dn = MFH_elec.converge(density.dm)
    print(MFH_elec.Etot)

    # Central region is a repetition of the electrodes without PBC
    HC = H_elec.tile(3,axis=0)
    HC.set_nsc([1,1,1])

    # Map electrodes in the device region, i.e., extract the atomic indices that correspond
    # to the electrodes inside the device region
    elec_indx = [range(len(H_elec)), range(len(HC.H)-len(H_elec), len(HC.H))]

    # MFH object of the central region
    MFH_HC = hh.HubbardHamiltonian(HC, U=U, kT=kT)

    # First create NEGF object, where we pass the MFH converged electrodes and the central region HubbardHamiltonian object
    negf = NEGF(MFH_HC, [MFH_elec, MFH_elec], elec_indx, elec_dir=['-A', '+A'])
    
    # Converge using Green's function method to obtain the densities
    dn = MFH_HC.converge(negf.dm_open, steps=1)

Optionally, we can also compute the transmission function using `TBtrans <https://launchpad.net/siesta>`_.
