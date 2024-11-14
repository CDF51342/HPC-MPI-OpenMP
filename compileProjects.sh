#!/bin/bash

# Check if a directory name is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Directory to process
directory=$1

# Check if the directory is valid
if [[ ! " CAP2024 OpenMP MPI MPI+OpenMP " =~ " $directory " ]]; then
    echo "Invalid directory: $directory"
    exit 1
fi

cd ./$directory
rm -r build
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=mpicxx.mpich
make
cd ../..
pwd
