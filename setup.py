#!/usr/bin/env python3

import multiprocessing
# pkg_resources is part of setuptools
import pkg_resources
import setuptools
from setuptools import find_packages, setup

packages = find_packages(include=["hubbard", "hubbard.*"])

metadata = dict(
    name="hubbard",
    platforms="any",

    # specify setuptools_scm version creation
    # TODO update this together with pyproject.toml
    setup_requires=[
        "setuptools>=46.0",
        "wheel",
        "setuptools_scm>=6.2",
    ],

    # Options should be specified in pyproject.toml
    use_scm_version={'fallback_version': '0.0.0.dev+$Format:%H$'},

    # This forces MANIFEST.in usage
    include_package_data=True,
    packages=packages,
)


if __name__ == "__main__":
    # Freeze to support parallel compilation when using spawn instead of fork
    multiprocessing.freeze_support()
    setup(**metadata)
