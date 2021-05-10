
Examples
========

These examples will show how to use the `hubbard` package to solve the mean-field Hubbard (MFH) equation to simulate electron interactions in graphene nanostructures:

.. math::
	H_{MFH} = H_0 + \sum_{i} U_i \left(\left\langle n_{i\downarrow}\right\rangle n_{i\uparrow}+\left\langle n_{i\uparrow}\right\rangle n_{i\downarrow}\right)- U\sum_i \left\langle n_{i\uparrow}\right\rangle\left\langle n_{i\downarrow}\right\rangle

Where :math:`H_0` is the pure tight-binding (TB) Hamiltonian of the structure.

One can also go through the `examples <https://github.com/dipc-cc/hubbard/tree/master/examples>`_ section in the main repository to explore more systems suitable to solve with the hubbard package.

.. toctree::
   :maxdepth: 1

   examples/molecules.ipynb
   examples/periodic.ipynb
   examples/open.ipynb
