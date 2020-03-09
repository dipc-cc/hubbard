#!/bin/bash


FILES=*.rst
for f in $FILES
do
  filename="${f%.*}"
  echo "Converting $f to ../../examples/$filename/README.md"
  `pandoc $f -f rst -t markdown -o ../../examples/$filename/README.md`
done
