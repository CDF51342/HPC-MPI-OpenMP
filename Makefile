# Define project directories
PROJECT1_DIR = ./CAP2024
PROJECT2_DIR = ./MPI
PROJECT3_DIR = ./MPI+OpenMP
PROJECT4_DIR = ./OpenMP

# Define build directories
BUILD_DIR1 = $(PROJECT1_DIR)/build
BUILD_DIR2 = $(PROJECT2_DIR)/build
BUILD_DIR3 = $(PROJECT3_DIR)/build
BUILD_DIR4 = $(PROJECT4_DIR)/build

# Targets to build each project
all: contrast_seq contrast_mpi contrast_mpi_omp contrast_omp

contrast_seq:
	@echo "Building Project 1..."
	mkdir -p $(BUILD_DIR1)
	cd $(BUILD_DIR1) && cmake .. -DCMAKE_CXX_COMPILER=mpicxx.mpich && $(MAKE)
	cp $(BUILD_DIR1)/contrast ./contrast_seq

contrast_mpi:
	@echo "Building Project 2..."
	mkdir -p $(BUILD_DIR2)
	cd $(BUILD_DIR2) && cmake .. -DCMAKE_CXX_COMPILER=mpicxx.mpich && $(MAKE)
	cp $(BUILD_DIR2)/contrast ./contrast_mpi

contrast_mpi_omp:
	@echo "Building Project 3..."
	mkdir -p $(BUILD_DIR3)
	cd $(BUILD_DIR3) && cmake .. -DCMAKE_CXX_COMPILER=mpicxx.mpich && $(MAKE)
	cp $(BUILD_DIR3)/contrast ./contrast_mpi_omp

contrast_omp:
	@echo "Building Project 4..."
	mkdir -p $(BUILD_DIR4)
	cd $(BUILD_DIR4) && cmake .. -DCMAKE_CXX_COMPILER=mpicxx.mpich && $(MAKE)
	cp $(BUILD_DIR4)/contrast ./contrast_omp

# Clean all build artifacts
clean:
	@echo "Cleaning all projects..."
	rm -rf $(BUILD_DIR1) $(BUILD_DIR2) $(BUILD_DIR3) $(BUILD_DIR4)
	rm -f contrast_seq contrast_mpi contrast_mpi_omp contrast_omp

.PHONY: all clean project1 project2 project3 project4
