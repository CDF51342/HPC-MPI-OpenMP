cmake_minimum_required(VERSION 3.10)

project(contrast VERSION 1.0 LANGUAGES CXX)

# Encontrar y enlazar MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    set(CMAKE_CXX_COMPILER mpic++)
    message(STATUS "MPI Found: ${MPI_CXX_LIBRARIES}")
    include_directories(${MPI_INCLUDE_PATH})
endif()

# Encontrar y enlazar OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "OpenMP Found: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Crear el ejecutable
add_executable(contrast contrast-enhancement.cpp histogram-equalization.cpp contrast.cpp)

# Enlazar automáticamente MPI y OpenMP
target_link_libraries(contrast MPI::MPI_CXX OpenMP::OpenMP_CXX)