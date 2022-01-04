#!/bin/sh
rm -r ~/.local/lib/python3.*/site-packages/hubbard*
python3 -m pip install . --prefix=~/.local
rm -r build
