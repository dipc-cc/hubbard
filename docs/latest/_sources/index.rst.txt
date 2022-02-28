.. hubbard documentation master file, created by
   sphinx-quickstart on Mon Oct 22 22:30:07 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: hubbard Package

Welcome to hubbard's documentation!
===================================

The hubbard_ Python package allows to find the self-consistent solution for the mean-field Hubbard Model
for a certain tight-binding Hamiltonian and a certain Coulomb repulsion parameter :math:`U`.

.. math::
      H &= -\sum_{ij\sigma}t_{ij\sigma}c^{\dagger}_{i\sigma}c_{j\sigma} + \sum_{i}U_in_{i\uparrow}n_{i\downarrow} +
      \frac{1}{2}\sum_{i\neq j\sigma\sigma^\prime}U_{ij}n_{i\sigma}n_{j\sigma^\prime} \approx\\
      & -\sum_{ij\sigma}t_{ij\sigma}c^{\dagger}_{i\sigma}c_{j\sigma} +
      \sum_{i\sigma} U_i \left\langle n_{i\sigma}\right\rangle n_{i\bar{\sigma}} +
      \frac{1}{2}\sum_{i\neq j\sigma}\left(U_{ij}
      + U_{ji}\right)\left(\langle n_{i\uparrow}\rangle + \langle n_{i\downarrow}\rangle\right)n_{j\sigma} + E_U


Where it has been sepparated into intra- (:math:`U_{i}`) and inter-atomic (:math:`U_{ij}`) Coulomb repulsion terms,
:math:`\langle n_{i\sigma}\rangle` is the :math:`\sigma=\uparrow,\downarrow`-spin density on site :math:`i` and :math:`E_U` is a constant term:

.. math::
         E_U = -\sum_i U_i \langle n_{i\uparrow}\rangle\langle n_{i\downarrow}\rangle -
         \frac{1}{2}\sum_{i\neq j}U_{ij}\left(\langle n_{i\uparrow}\rangle+\langle n_{i\downarrow}\rangle\right)\left(\langle n_{j\uparrow}\rangle
         + \langle n_{j\downarrow}\rangle\right)

which can be directly added to the total electronic energy.

This package allows for_

   * Easy calculations of spin-resolved quantities.
      It takes advantage of many routines from sisl_ as well as numpy_ and scipy_,
      which makes it very efficient when handling with thousands of atoms, given the usage of sparse matrices.
      The goal of this package is to include electron correlations in the tight-binding Hamiltonian by solving self-consistently
      the mean-field Hubbard model. Given the simplicity of the model one can find the solution in short time to problems that are typically
      adressed with DFT and obtain similar accuracy, especially for sp2 carbon systems. Here it is also very easy to manipulate
      the spin configuration to obtain different magnetic solutions, e.g., obtain the approximated energy difference between the singlet and
      the triplet states, etc. This package is fully implemented in Python, which makes it very easy and comfortable to use.


   * It provides with nice plotting functions to visualize the different physical quantities
      that are obtained with the hubbard package,
      such as the spin-polarization, wavefunctions for each spin-channel, density of states maps, etc.


.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   examples.rst

.. toctree::
   :maxdepth: 2
   :caption: Publications

   publications

.. toctree::
   :maxdepth: 2
   :caption: Funding

   funding

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Contributing
============

Contributions are highly appreciated.

-  If you find any bugs plase form a `bug report/issue <https://github.com/dipc-cc/hubbard/issues>`_.

-  If you have a fix please consider adding a `pull request <https://github.com/dipc-cc/hubbard/pulls>`_.


Indices and tables
==================

The complete package index can be found below:

* :ref:`modindex`
* :ref:`genindex`

.. _sisl: https://sisl.readthedocs.io/en/latest/introduction.html
.. _numpy: https://numpy.org/
.. _scipy: https://www.scipy.org/
.. _hubbard: https://github.com/dipc-cc/hubbard
