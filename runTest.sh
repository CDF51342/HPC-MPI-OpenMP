#!/bin/bash
#SBATCH --job-name=contrast_job
#SBATCH --output=contrast_output.txt
#SBATCH --error=contrast_error.txt
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --partition=compute

versions=("OpenMP" "MPI" "MPI+OpenMP")

read -p "Do you want to compile the projects? (y/n): " compile_choice
if [ "$compile_choice" == "y" ]; then
    printf "Compiling projects...\n"
    bash compileProjects.sh
    printf "Projects compiled\n"
else
    printf "Skipping compilation...\n"
fi

printf "Executing sequential version...\n"
mpirun -np 1 ./CAP2024/build/contrast
# Save the output of the sequential version to check the parallel versions
mkdir OutputCheck
mv ./out* ./OutputCheck

# Execute OpenMP version, and save the output
mpirun -np 1 ./OpenMP/build/contrast
# check the difference between the sequential and OpenMP versions
check_output_difference() {
    version=${versions[$1]}
    passed=true
    for file in ./out*; do
        diff "$file" "./OutputCheck/$(basename "$file")" > /dev/null
        if [ $? -ne 0 ]; then
            echo -e "\e[31m✘\e[0m Error: $file differs from OutputCheck/$(basename "$file") for $version version"
            passed=false
        fi
    done
    if [ "$passed" = true ]; then
        echo -e "\e[32m✔\e[0m $version version passed!"
        passed_versions[$1]="1"
    fi
}

passed_versions=("0" "0" "0")

check_output_difference 0

# Execute MPI version, and save the output
mpirun -np 4 ./MPI/build/contrast
# check the difference between the sequential and MPI versions
check_output_difference 1

# Execute MPI+OpenMP version, and save the output
mpirun -np 4 ./MPI+OpenMP/build/contrast
# check the difference between the sequential and MPI+OpenMP versions
check_output_difference 2

rm ./out*

# End of script
rm -r OutputCheck

# Summary of passed versions
echo "==================================="
echo "Summary of passed versions:"
for i in "${!versions[@]}"; do
    version=${versions[$i]}
    if [ "${passed_versions[$i]}" == "0" ]; then
        echo -e "\e[31m✘\e[0m Error: $version version"
    else
        echo -e "\e[32m✔\e[0m $version version passed!"
    fi
done