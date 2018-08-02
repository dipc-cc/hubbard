[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

# Hubbard #

__Hubbard__ is a Python package for solving the meanfield Hubbard model.

This project was initiated by Sofia Sanz and Thomas Frederiksen at DIPC in 2018.


## Dependencies ##
Before installation of __Hubbard__ the following packages are required
   - numpy >= 1.8
   - netCDF4
   - matplotlib
   - sisl

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
__Hubbard__ is distributed under [LGPL][lgpl], please see the LICENSE file.


<!---
Links to external and internal sites.
-->
[issues]: https://github.com/dipc-cc/hubbard/issues
[pulls]: https://github.com/dipc-cc/hubbard/pulls
[lgpl]: http://www.gnu.org/licenses/lgpl.html
