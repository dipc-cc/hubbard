[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

# hubbard #

__hubbard__ is a Python package for solving the mean-field Hubbard model built on [sisl].

This project was initiated by Sofia Sanz and Thomas Frederiksen at DIPC in 2018.


## Dependencies ##
Before installation of __hubbard__ the following packages are required
   - python >= 3.6
   - numpy >= 1.13
   - scipy >= 0.18
   - netCDF4 >= 1.3.1
   - matplotlib >= 2.2.2
   - [sisl][sisl] >= 0.9.9

## Installation ##
Manual installation is performed with the command

    python setup.py install --prefix=<prefix>
    # or
    python setup.py install --home=<my-python-home>

One may also wish to set the following environment variables

    export PYTHONPATH=<my-python-home>/lib/python/
    export PATH=$PATH:<my-python-home>/bin/

## Contributions, issues and bugs ##
Contributions are highly appreciated.

If you find any bugs please form a [bug report/issue][issues]

If you have a fix please consider adding a [pull request][pulls].

## License ##
__hubbard__ is distributed under [LGPL][lgpl], please see the LICENSE file.


## Funding ##
Financial support from Spanish AEI (FIS2017-83780-P) and the European Union's Horizon 2020 (FET-Open
project [SPRING][spring] Grant No. 863098) is acknowledged.

<!---
Links to external and internal sites.
-->
[issues]: https://github.com/dipc-cc/hubbard/issues
[pulls]: https://github.com/dipc-cc/hubbard/pulls
[lgpl]: http://www.gnu.org/licenses/lgpl.html
[sisl]: https://github.com/zerothi/sisl
[spring]: https://www.springfetopen.eu/
