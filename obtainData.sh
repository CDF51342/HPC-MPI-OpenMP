#!/bin/bash

# Se compila todas las versiones
make

# Se obtienen los datos de OpenMP
schedule="static guided dynamic"
chunk_size="16 32 64 128 256 512 1024 2048 4096"
num_threads="1 2 4 8 16"

for n in $num_threads; do
    export OMP_NUM_THREADS=$n
    for sche in $schedule; do
        export C_OMP_SCHEDULE="$sche"
        for chunk in $chunk_size; do
            export C_OMP_CHUNK_SIZE=$chunk
            for i in $(seq 1 5); do
                srun -p gpus -N 1 -n 1 ./contrast_omp
            done
        done
    done
done


# Se obtienen los datos de MPI
# Primero los de 1 nodo, debido a que no puede hacer 16 procesos
processes="1 2 3 4 8"
node="1"

for process in $processes; do
    for i in $(seq 1 5); do
        mkdir -p data/MPI/Node_"$node"
        srun -p gpus -N "$node" -n "$process" ./contrast_mpi > data/MPI/Node_"$node"/output_p"$process"_"$i".csv
    done
done

# Ahora se hacen para 2, 3 y 4 nodos hasta 16 procesos
processes="1 2 3 4 8 16"
node="2 3 4"

for nod in $node; do
    for process in $processes; do
        for i in $(seq 1 5); do
            mkdir -p data/MPI/Node_"$nod"
            srun -p gpus -N "$nod" -n "$process" ./contrast_mpi > data/MPI/Node_"$nod"/output_p"$process"_"$i".csv
        done
    done
done

# Se obtienen los datos de la versión híbrida
# Se establecen el schedule y chunk size para todas las pruebas
export C_OMP_SCHEDULE="guided"
export C_OMP_CHUNK_SIZE="64"
num_threads="1 2 4 8 16"

# Primero los de 1 nodo, debido a que no puede hacer 16 procesos
processes="1 2 3 4 8"
node="1"

for process in $processes; do
    for n in $num_threads; do
        export OMP_NUM_THREADS=$n
        for i in $(seq 1 5); do
            mkdir -p data/MPI+OpenMP/Node_"$node"
            srun -p gpus -N "$node" -n "$process" ./contrast_mpi_omp > data/MPI+OpenMP/Node_"$node"/output_p"$process"_threads"$n"_"$i".csv
        done
    done
done

# Ahora se hacen para 2, 3 y 4 nodos hasta 16 procesos
processes="1 2 3 4 8 16"
node="2 3 4"

for nod in $node; do
    for process in $processes; do
        for i in $(seq 1 5); do
            mkdir -p data/MPI+OpenMP/Node_"$nod"
            srun -p gpus -N "$nod" -n "$process" ./contrast_mpi_omp > data/MPI+OpenMP/Node_"$nod"/output_p"$process"_threads"$n"_"$i".csv
        done
    done
done

# Se eliminan los ejecutables
make clean
