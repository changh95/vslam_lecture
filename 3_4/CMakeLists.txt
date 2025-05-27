cmake_minimum_required(VERSION 3.10)
project("bag_of_words" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "Found OpenCV library: " ${OpenCV_INCLUDE_DIRS})
    include_directories(${OpenCV_INCLUDE_DIRS})
endif (OpenCV_FOUND)

find_package(DBoW2 REQUIRED)
if (DBoW2_FOUND)
    message(STATUS "Found DBoW2 library")
    include_directories(${DBoW2_INCLUDE_DIRS})
endif (DBoW2_FOUND)

add_executable(match_bow examples/match_bow.cpp)
target_link_libraries(match_bow 
    ${OpenCV_LIBS}
    ${DBoW2_LIBS}
)
