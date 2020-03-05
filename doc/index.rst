.. Hubbard documentation master file, created by
   sphinx-quickstart on Mon Oct 22 22:30:07 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: Hubbard Package

Welcome to Hubbard's documentation!
===================================

The hubbard package allows to find the self-consistent solution for the Mean Field Hubbard Model 
for a certain tight-binding Hamiltonian and a certain Coulomb repulsion parameter U

-  Easy calculation of spin-resolved quantities. It takes advantage of many routines from sisl_, 
   which makes it very efficient when handling with thousands of atoms, given the usage of sparse matrices. 
   The goal of this package is to include electron correlations in the tight-binding Hamiltonian by solving self-consistently 
   the Mean Field Hubbard model. Given the simplicity of the model one can find quite fast the solution to problems that are typically 
   adressed with DFT with similar accuracy, specially for sp2 carbon systems. It is also very easy to manipulate 
   the spin configuration to obtain different magnetic solutions, e.g., obtain the approximated energy difference between the singlet and 
   the triplet states, etc. This package is fully implemented in Python, which makes it very easy and comfortable to use.

-  It provides with nice plotting functions to visualize the different physical quantities that are obtained with the hubbard package, 
   such as the spin-polarization, wavefunctions for each spin-channel, the density of states maps, etc.

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation

.. toctree::
   :maxdepth: 2
   :caption: Publications

   publications

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   api

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _sisl: https://sisl.readthedocs.io/en/latest/introduction.html
