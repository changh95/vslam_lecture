cmake_minimum_required(VERSION 3.10)
project("orb feature matching" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "Found OpenCV library: " ${OpenCV_INCLUDE_DIRS})
    include_directories(${OpenCV_INCLUDE_DIRS})
endif (OpenCV_FOUND)

add_executable(detect_feature examples/detect_feature.cpp)
target_link_libraries(detect_feature ${OpenCV_LIBS})