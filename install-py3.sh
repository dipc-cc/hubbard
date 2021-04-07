rm -r ~/.local/lib/python3.*/site-packages/hubbard*
python3 setup.py install --prefix=~/.local
rm -r build
