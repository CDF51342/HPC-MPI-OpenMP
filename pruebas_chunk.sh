schedule="static guided dynamic"
chunk_size="8 16 32 64 128"

for chunk in $chunk_size; do
    export C_OMP_CHUNK_SIZE=$chunk
    for sche in $schedule; do
        export C_OMP_SCHEDULE="$sche"

        for i in $(seq 1 6); do
            ./contrast_omp
        done
    done
done
