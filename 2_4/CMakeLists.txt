cmake_minimum_required(VERSION 3.10)
project(slam_2_4)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})

# Create executables
add_executable(eigen_example examples/eigen.cpp)
add_executable(rotations_example examples/rotations.cpp)

# Link libraries
target_link_libraries(eigen_example Eigen3::Eigen)
target_link_libraries(rotations_example 
    Eigen3::Eigen
    Sophus::Sophus
)
