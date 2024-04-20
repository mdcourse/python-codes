#!/bin/bash
set -e

# pull the last version
git pull

# update lammps tutorials in case changes were made
git submodule update --remote

jupyter nbconvert --to script build-documentation.ipynb
python3 build-documentation.py
rm build-documentation.py