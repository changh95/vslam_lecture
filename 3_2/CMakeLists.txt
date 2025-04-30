cmake_minimum_required(VERSION 3.10)
project(feature_extraction)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Add executables
add_executable(detect_feature detect_feature.cpp)
target_link_libraries(feature_extraction ${OpenCV_LIBS})
