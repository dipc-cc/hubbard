#!/bin/bash

#sudo apt install python3-sphinx python3-numpydoc python3-sphinx-rtd-theme
rm -rf api-gen latest
make clean
sphinx-apidoc -fMeET -o api-gen ../Hubbard ../Hubbard/**/setup.py
make html
rm -r _build/html/_sources
mv _build/html latest
rm -rf api-gen _build _static _templates
