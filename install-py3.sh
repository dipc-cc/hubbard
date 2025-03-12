#!/bin/sh
rm -r ~/.local/lib/python3.*/site-packages/hubbard*
python3 -m pip install . --user
rm -r build
