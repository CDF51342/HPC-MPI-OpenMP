# Directories to process
directories=("CAP2024" "OpenMP" "MPI" "MPI+OpenMP")

for dir in "${directories[@]}"; do
    cd ./$dir
    rm -r build
    mkdir build
    cd build
    cmake .. -DCMAKE_CXX_COMPILER=mpicxx.mpich
    make
    cd ../..
    pwd
done

