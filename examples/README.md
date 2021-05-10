# hubbard Examples #

This section contains some examples solved using the `hubbard` package:

## Simulation of electron correlations in a molecule ##

[kondo-paper][example-kondo] contains the molecular geometry in `.XV` format and `python` scripts to reproduce figures in Ref.  [Nature Communications 10, 200 (2019)](https://www.nature.com/articles/s41467-018-08060-6)

[clar-goblet][clar-goblet] contains the molecular geometry in `.xyz` format and a `python` script to obtain the self-consistent solution for this molecule with the mean-field Hubbard approximation

[anthracenes][anthracenes] contains the molecular geometry in `.XV` format of antrhacenes of width 2,3 and 4 benzene rings and a `python` script to obtain the self-consistent solution for these molecules with the mean-field Hubbard approximation

## Simulation of electron correlations in a periodic system ##

[This example][example-periodic] contains a python script that computes the self-consistent solution of a periodic 1D armchair and zigzag nanorribons along the x-direction. The script benchmarks [Ref. Phys. Rev. B 81, 245402 (2010)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.245402)


## Simulation of electron correlations in open-quantum system ##

[AGNR-constriction][example-open-AGNR] contains a python script to compute the self-consistent solution for a 1D non-periodic armchair graphene nanorribon along the x-direction with a scattering area (defect). It also contains an input file  (`RUN.fdf`) for [TBtrans](https://gitlab.com/siesta-project/siesta) to calculate the transmission probability through the nanoconstriction per spin-channel.

[ZGNR-constriction][example-open-ZGNR] contains a python script to compute the self-consistent solution for a 1D non-periodic zigzag graphene nanorribon along the x-direction with a scattering area (defect). It also contains an input file (`RUN.fdf`) for [TBtrans](https://gitlab.com/siesta-project/siesta) to calculate the transmission probability through the nanoconstriction per spin-channel.

These open quantum systems are solved using the `hubbard.NEGF` class. The script benchmarks [Ref. Phys. Rev. B 81, 245402 (2010)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.245402)

<!---
Links to external and internal sites.
-->
[example-kondo]: https://github.com/dipc-cc/hubbard/tree/master/examples/molecules/kondo-paper
[anthracenes]: https://github.com/dipc-cc/hubbard/tree/master/examples/molecules/anthracenes
[clar-goblet]: https://github.com/dipc-cc/hubbard/tree/master/examples/molecules/clar-goblet
[example-periodic]: https://github.com/dipc-cc/hubbard/tree/master/examples/periodic/benchmarks
[example-open-AGNR]: https://github.com/dipc-cc/hubbard/tree/master/examples/open/benchmarks/AGNR-constriction
[example-open-ZGNR]: https://github.com/dipc-cc/hubbard/tree/master/examples/open/benchmarks/ZGNR-constriction
