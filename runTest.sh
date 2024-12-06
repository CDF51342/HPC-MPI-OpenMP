#!/bin/bash
#SBATCH --job-name=contrast_job
#SBATCH --output=contrast_output.txt
#SBATCH --error=contrast_error.txt
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --partition=compute

versions=("OpenMP" "MPI" "MPI+OpenMP")

echo "=================================================================="
# Print parameter
if [ -z "$1" ]; then
    echo "Running all versions"
else
    echo -e "Running version \e[35m${versions[$1]}\e[0m"
fi
echo "=================================================================="

echo "Recompiling..."
make -B > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "\e[31m✘\e[0m Error: Compilation failed"
    exit 1
fi

rm ./out* -f

# Check if images_seq directory exists
if [ ! -d "./images_seq" ]; then
    printf "Executing sequential version...\n"
    mpirun -np 1 ./CAP2024/build/contrast
    # Save the output of the sequential version to check the parallel versions
    mkdir images_seq
    mv ./out* ./images_seq
    echo "=================================================================="
fi

# Execute OpenMP version, and save the output
run_openmp() {
    mpirun -np 1 ./OpenMP/build/contrast
    check_output_difference 0
}

# Execute MPI version, and save the output
run_mpi() {
    mpirun -np 4 ./MPI/build/contrast
    check_output_difference 1
}

# Execute MPI+OpenMP version, and save the output
run_mpi_openmp() {
    mpirun -np 4 ./MPI+OpenMP/build/contrast
    check_output_difference 2
}

# check the difference between the sequential and parallel versions
check_output_difference() {
    echo "==================================="
    version=${versions[$1]}
    echo "Checking output difference for $version version..."
    passed=true
    for file in ./images_seq/out*; do
        diff "$(basename "$file")" "./images_seq/$(basename "$file")" > result.txt 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\e[31m✘\e[0m Error: $(basename "$file") differs from images_seq/$(basename "$file") for $version version"
            passed=false
        fi
    done
    if [ "$passed" = true ]; then
        echo -e "\e[32m✔\e[0m $version version passed!"
        passed_versions[$1]="1"
    fi
    echo "==================================="
}

passed_versions=("0" "0" "0")

if [ -z "$1" ]; then
    run_openmp
    run_mpi
    run_mpi_openmp
else
    case $1 in
        0) run_openmp ;;
        1) run_mpi ;;
        2) run_mpi_openmp ;;
        *) echo "Invalid parameter. Use 0 for OpenMP, 1 for MPI, or 2 for MPI+OpenMP." ;;
    esac
fi

# rm ./out*

if [ -z "$1" ]; then
    # Summary of passed versions
    echo "Summary of passed versions:"
    echo "==================================="
    for i in "${!versions[@]}"; do
        version=${versions[$i]}
        if [ "${passed_versions[$i]}" == "0" ]; then
            echo -e "\e[31m✘\e[0m Error: $version version"
        else
            echo -e "\e[32m✔\e[0m $version version passed!"
        fi
    done
    echo "==================================="
fi
