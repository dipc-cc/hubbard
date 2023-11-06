[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10058802.svg)](https://doi.org/10.5281/zenodo.10058802)

# hubbard #

__hubbard__ is a Python package built on [sisl][sisl] for solving the mean-field Hubbard model.

This project was initiated by Sofia Sanz and Thomas Frederiksen at DIPC in 2018.

Recommended [online documentation][doc].

## Dependencies ##
Before installation of __hubbard__ the following packages are required
   - python >= 3.6
   - numpy >= 1.13
   - scipy >= 0.18
   - matplotlib >= 2.2.2
   - [sisl][sisl] >= 0.12.1

Optional dependencies:
   - [netCDF4][netcdf] >= 1.3.1

## Installation ##
Manual installation is performed with the command

    python3 -m pip install . --prefix=<prefix>


## Contributions, issues and bugs ##
Contributions are highly appreciated.

If you find any bugs please form a [bug report/issue][issues].

If you have a fix please consider adding a [pull request][pulls].


## License ##
__hubbard__ is distributed under [LGPL][lgpl], please see the LICENSE file.


## Funding ##
Financial support from Spanish AEI (FIS2017-83780-P, PID2020-115406GB-I00), the Basque Departamento de Educación through the PhD scholarship no. PRE_2020_2_0049 (S.S.) and the European Union's Horizon 2020 (FET-Open project [SPRING][spring] Grant No. 863098) is acknowledged.


<!---
Links to external and internal sites.
-->
[issues]: https://github.com/dipc-cc/hubbard/issues
[pulls]: https://github.com/dipc-cc/hubbard/pulls
[lgpl]: http://www.gnu.org/licenses/lgpl.html
[sisl]: https://github.com/zerothi/sisl
[spring]: https://www.springfetopen.eu/
[doc]: https://dipc-cc.github.io/hubbard/docs/latest/index.html
[netcdf]: https://github.com/Unidata/netcdf4-python
