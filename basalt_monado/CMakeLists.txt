cmake_minimum_required(VERSION 3.10...3.18)

project(basalt)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Try to use mold linker if found
find_program(MOLD_PROGRAM mold)
if(MOLD_PROGRAM)
  message(STATUS "Mold linker found: ${MOLD_PROGRAM}")
  set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=mold")
  set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=mold")
else()
  message(STATUS "Mold linker not found. Using default linker.")
endif()

if(NOT EIGEN_ROOT)
  set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/basalt-headers/thirdparty/eigen")
endif()

string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWERCASE)
find_program(DPKG_PROGRAM dpkg DOC "dpkg program of Debian-based systems")
if(DPKG_PROGRAM)
  execute_process(
    COMMAND ${DPKG_PROGRAM} --print-architecture OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif(DPKG_PROGRAM)

find_program(LSB_RELEASE_PROGRAM lsb_release DOC "lsb_release program of Debian-based systems")
if(LSB_RELEASE_PROGRAM)
  execute_process(
    COMMAND ${LSB_RELEASE_PROGRAM} -rs OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(${LSB_RELEASE_ID_SHORT} EQUAL "22.04")
    set(DEBIAN_DEPENDS
        "libtbb12, liblz4-1, libbz2-1.0, libboost-filesystem1.74.0, libboost-date-time1.74.0, libboost-program-options1.74.0, libboost-regex1.74.0, libopencv-dev, libglew2.2, libjpeg8, libpng16-16, libuvc0, libfmt8, libc++-14-dev, libc++1-14, libc++abi1-14, libunwind-14, libunwind-14-dev, libwayland-bin, libepoxy-dev"
    )
  endif()

endif(LSB_RELEASE_PROGRAM)

string(TIMESTAMP PROJECT_VERSION_REVISION "%Y%m%d%H%M")

# Get VIT interface version. Note that modifying vit_interface.h won't trigger
# the regeneration of basalt.map by itself.
set(VIT_INTERFACE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vit/vit_interface.h)
file(READ ${VIT_INTERFACE_FILE} VIT_INTERFACE_CONTENTS)
string(REGEX MATCH "#define VIT_HEADER_VERSION_MAJOR ([0-9]+)" _ ${VIT_INTERFACE_CONTENTS})
set(VIT_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define VIT_HEADER_VERSION_MINOR ([0-9]+)" _ ${VIT_INTERFACE_CONTENTS})
set(VIT_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define VIT_HEADER_VERSION_PATCH ([0-9]+)" _ ${VIT_INTERFACE_CONTENTS})
set(VIT_VERSION_PATCH ${CMAKE_MATCH_1})

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Mateo de Mayo <mateo.demayo@collabora.com>")
set(CPACK_PACKAGE_VERSION_MAJOR ${VIT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VIT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH "${VIT_VERSION_PATCH}-${PROJECT_VERSION_REVISION}~${LSB_RELEASE_ID_SHORT}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS ${DEBIAN_DEPENDS})
set(CPACK_PACKAGE_FILE_NAME
    "${PROJECT_NAME_LOWERCASE}_${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}"
)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
include(CPack)

# Configure CCache if available
if(NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(STATUS "Dind't find ccache")
  endif()
else()
  message(STATUS "Compiler launcher already set. Not configuring ccache.")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CXX_MARCH)
  set(CXX_MARCH native)
endif()

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(BASALT_MARCH_FLAGS "-march=${CXX_MARCH}")
else()
  message(STATUS "Running on Apple ${CMAKE_SYSTEM_PROCESSOR}. Disabled -march flag.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Flags used for CHECK_CXX_SOURCE_COMPILES
set(CMAKE_REQUIRED_FLAGS "-Wno-error")

# save flags passed by user
set(BASALT_PASSED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN") # cmake default: "-g"
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN") # cmake default: "-O2 -g -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG") # cmake default: "-O3 -DNDEBUG"
set(CMAKE_CXX_FLAGS_CIDEBUG "-O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN") # CI version with no debug symbols
set(CMAKE_CXX_FLAGS_CIRELWITHDEBINFO "-O3 -DEIGEN_INITIALIZE_MATRICES_BY_NAN") # CI version with no debug symbols

# Limit number of jobs.
# As of now, Basalt compilation requires >16gb if you want to use >6 jobs.
execute_process(
  COMMAND bash -c "grep -m1 -oP 'MemAvailable:\\s+\\K[0-9]+' /proc/meminfo" OUTPUT_VARIABLE TOTAL_MEMORY
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TOTAL_MEMORY LESS 16000000)
  set(CMAKE_BUILD_PARALLEL_LEVEL 6 CACHE STRING "Number of parallel build processes")
endif()

# base set of compile flags
set(BASALT_CXX_FLAGS "-Wall -Wextra -ftemplate-backtrace-limit=0")

# For avoiding libfmt >9 issues, see https://github.com/strasdat/Sophus/issues/366#issuecomment-1229178088
add_definitions(-DFMT_DEPRECATED_OSTREAM=1)

# clang-specific compile flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -fcolor-diagnostics")

  #   - Added TBB_USE_GLIBCXX_VERSION macro to specify the version of GNU
  #     libstdc++ when it cannot be properly recognized, e.g. when used
  #     with Clang on Linux* OS. Adopted from https://github.com/wjakob/tbb
  if(NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    endif()
    add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -fdiagnostics-color=always")
endif()

# Set platform / compiler specific compile flags and checks
if(APPLE)
  # Need to investigate how to reliably detect and use OpenMP on macOS...
  #  set(USE_OPENMP_DEFAULT OFF)

  # Among others, setting CMAKE_FIND_FRAMEWORK to LAST fixed issues
  # with installed Mono that contains old headers (libpng, ...).
  # See: https://github.com/openMVG/openMVG/issues/1349#issuecomment-401492811
  set(CMAKE_FIND_FRAMEWORK LAST)

  if(CMAKE_SYSTEM_VERSION VERSION_LESS 19.0.0)
    # use brewed llvm's libc++
    include_directories("/usr/local/opt/llvm/include/c++/v1")
    link_directories("/usr/local/opt/llvm/lib")
    add_compile_options("-nostdinc++")
    #set(STD_CXX_FS c++fs)

    # Workaround for cmake not to filter the manually added standard include path
    # See: https://gitlab.kitware.com/cmake/cmake/issues/19227#note_669894
    list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/usr/local/opt/llvm/include/c++/v1")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Detected macOS with non-Apple clang")

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Detected macOS with Apple clang")
    # Apple clang on macOS < 10.14 Mojave is too old
    if(CMAKE_SYSTEM_VERSION VERSION_LESS 18.0.0)
      message(
        WARNING
          "Detected Darwin version ${CMAKE_SYSTEM_VERSION}, which is earlier than macos 10.14 Mojave. Apple clang is too old and not supported. Use clang from homebrew."
      )
    endif()

  else()
    message(WARNING "Detected macOS with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()

elseif(UNIX)
  # assume libstdc++
  set(STD_CXX_FS stdc++fs)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Detected Linux with clang.")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Detected Linux with gcc.")
  else()
    message(WARNING "Detected Linux with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()

elseif(WIN32)
  add_compile_options(-Wno-c++11-narrowing)
  add_compile_options(-Wno-deprecated)
endif()

# OpenMP option and compile flags
#
# Note: OpenMP and TBB don't mix well, so we disable Eigen's parallelization.
# It's trying to paralellize matrix products during SC, which we run in a parallel_reduce using TBB.
# Turns out using OpenMP can slow down the computby factor 10-100x! So for now we discable it completely.
# One place where Eigen's parallelization could still have been useful is the CG solver in the mapper.
# We could in the future investiagte other implementations (paralellized with TBB) or selectively enabling
# Eigen's parallelization just for CG, setting number of threads to 1 everywhere else.
# Another way to ensure Eigen doesn't use OpenMP regardless of how it was built is setting the environment
# variable OMP_NUM_THREADS=1 beofre running the application.
#
# See: https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
#
# If we enable BLAS / LAPACK either directly or via thirdparty libs like ceres,
# make sure to disable OpenMP for the linked BLAS library. In particular on Ubuntu it seems OpenBLAS is often installed,
# and it can have similar issues in multithreaded applications if it's own parallelization with OpenMP is enabled.
# You can set the environment varaibles OPENBLAS_NUM_THREADS=1 or OMP_NUM_THREADS=1. This is also mentioned in the ceres
# installation documentation.
#
# See also: https://github.com/xianyi/OpenBLAS/wiki/faq#multi-threaded
#
# Set EIGEN_DONT_PARALLELIZE to be sure it doesn't use openmp,
# just in case some dependency enables openmp without us noticing.
set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")

# Make specialization for double / float optional.
# Disabling them can be useful for better compile times during development.
option(BASALT_INSTANTIATIONS_DOUBLE "Instatiate templates for Scalar=double." ON)
option(BASALT_INSTANTIATIONS_FLOAT "Instatiate templates for Scalar=float." ON)
option(BASALT_BUILD_SHARED_LIBRARY_ONLY "Build only Basalt shared library" OFF)

if(BASALT_BUILD_SHARED_LIBRARY_ONLY)
  if(NOT DEFINED BASALT_INSTANTIATIONS_DOUBLE)
    set(BASALT_INSTANTIATIONS_DOUBLE OFF)
  endif()
  if(NOT DEFINED BUILD_TESTS)
    set(BUILD_TESTS OFF)
  endif()
endif()

if(BASALT_INSTANTIATIONS_DOUBLE)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_DOUBLE)
endif()
if(BASALT_INSTANTIATIONS_FLOAT)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_FLOAT)
endif()

# setup combined compiler flags
set(CMAKE_CXX_FLAGS "${BASALT_CXX_FLAGS} ${BASALT_MARCH_FLAGS} ${BASALT_PASSED_CXX_FLAGS}")

set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.4.0 EXACT REQUIRED MODULE)
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")
if(NOT EIGEN3_INCLUDE_DIR MATCHES "^${EIGEN_ROOT}")
  message(WARNING "Found Eigen headers are outside of specified EIGEN_ROOT '${EIGEN_ROOT}'")
endif()

if(WIN32)
  find_package(TBB CONFIG REQUIRED)
else()
  find_package(TBB REQUIRED)
endif()

message(
  STATUS
    "Found TBB ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} (interface version ${TBB_INTERFACE_VERSION}) headers in: ${TBB_INCLUDE_DIRS}"
)
if(TBB_INTERFACE_VERSION LESS 11004)
  # enable global_control header for earlier TBB versions (Ubuntu 16.04, 18.04)
  add_definitions(-DTBB_PREVIEW_GLOBAL_CONTROL)
endif()

# NOTE: not specifying version, since 2, 3 or 4 is fine
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui)
message(STATUS "Found OpenCV ${OpenCV_VERSION} headers in: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV_LIBS: ${OpenCV_LIBS}")

# Add our own custom scoped opencv target since none is provided by OpenCV itself
add_library(basalt::opencv INTERFACE IMPORTED)
set_property(TARGET basalt::opencv PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET basalt::opencv PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})

find_package(fmt REQUIRED)
message(STATUS "Found {fmt} ${fmt_VERSION} in: ${fmt_DIR}")

add_subdirectory(thirdparty)

# custom scoped vit target: https://gitlab.freedesktop.org/monado/utilities/vit/
add_library(basalt::vit INTERFACE IMPORTED)
set_property(TARGET basalt::vit PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vit)
target_sources(basalt::vit INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vit/vit_implementation_helper.cpp)

# custom scoped cli11 target
add_library(basalt::cli11 INTERFACE IMPORTED)
set_property(
  TARGET basalt::cli11 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/CLI11/include
)

# custom scoped magic_enum target
add_library(basalt::magic_enum INTERFACE IMPORTED)
set_property(
  TARGET basalt::magic_enum PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                     ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/magic_enum/include
)

# main library
add_library(basalt_internal STATIC)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/build_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/basalt/utils/build_config.h
)
target_include_directories(basalt_internal PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

# List all header and source files with target_sources.
# This ensures, e.g., that QtCreator >= 6.0 properly recognize these headers to belong to the project.
# To support cmake < 3.13, use absolute paths (see: https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/)
target_sources(
  basalt_internal
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/hash_bow/hash_bow.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/block_diagonal.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/imu_block.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/landmark_block.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/landmark_block_abs_dynamic.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_abs_qr.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_abs_sc.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_base.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_rel_sc.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/frame_to_frame_optical_flow.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/multiscale_frame_to_frame_optical_flow.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/optical_flow.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/patch.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/patch_optical_flow.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/patterns.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/accumulator.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/linearize.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/poses_linearize.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/poses_optimize.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/spline_linearize.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/spline_optimize.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/ba_utils.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/cast_utils.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/common_types.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/filesystem.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/format.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/imu_types.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/keypoints.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/nfr.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/sim_utils.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/system_utils.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/test_utils.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/time_utils.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/tracks.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/union_find.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/vio_config.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/vis_matrices.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/vis_utils.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/ba_base.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/landmark_database.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/marg_helper.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sc_ba_base.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sqrt_ba_base.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sqrt_keypoint_vio.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sqrt_keypoint_vo.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/vio_estimator.h
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/io/marg_data_io.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/landmark_block.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_abs_qr.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_abs_sc.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_base.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_rel_sc.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/optical_flow/optical_flow.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/keypoints.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/system_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/time_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/vio_config.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/vis_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/ba_base.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/landmark_database.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/marg_helper.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sc_ba_base.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sqrt_ba_base.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sqrt_keypoint_vio.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sqrt_keypoint_vo.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/vio_estimator.cpp
)

target_link_libraries(
  basalt_internal
  PUBLIC ${STD_CXX_FS}
         basalt::opencv
         basalt::basalt-headers
         TBB::tbb
         pangolin
         basalt::cli11
         basalt::vit
         basalt::magic_enum
  PRIVATE nlohmann::json fmt::fmt
)
target_include_directories(basalt_internal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(basalt_internal PUBLIC ${BASALT_COMPILE_DEFINITIONS})
#target_compile_definitions(basalt_internal PUBLIC BASALT_DISABLE_ASSERTS)

if(NOT BASALT_BUILD_SHARED_LIBRARY_ONLY)
  target_sources(
    basalt_internal
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/aprilgrid.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/calibration_helper.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/cam_calib.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/cam_imu_calib.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/vignette.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/device/rs_t265.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_euroc.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_kitti.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_rosbag.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_uzh.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/marg_data_io.h
           ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/nfr_mapper.h
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/aprilgrid.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/calibraiton_helper.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/vignette.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/io/dataset_io.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/nfr_mapper.cpp
  )
  target_link_libraries(basalt_internal PRIVATE rosbag apriltag opengv)

endif()

add_library(basalt SHARED)

target_sources(
  basalt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vit/vit_tracker.hpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/vit/vit_tracker.cpp
)

target_compile_definitions(basalt PRIVATE VIT_INTERFACE_IMPLEMENTATION)

target_link_libraries(basalt basalt_internal)

if(NOT BASALT_BUILD_SHARED_LIBRARY_ONLY)

  add_executable(basalt_calibrate src/calibrate.cpp src/calibration/cam_calib.cpp)
  target_link_libraries(basalt_calibrate basalt_internal pangolin basalt::cli11)

  add_executable(basalt_calibrate_imu src/calibrate_imu.cpp src/calibration/cam_imu_calib.cpp)
  target_link_libraries(basalt_calibrate_imu basalt_internal pangolin basalt::cli11)

  add_executable(basalt_vio_sim src/vio_sim.cpp)
  target_link_libraries(basalt_vio_sim basalt_internal pangolin basalt::cli11)

  add_executable(basalt_mapper_sim src/mapper_sim.cpp)
  target_link_libraries(basalt_mapper_sim basalt_internal pangolin basalt::cli11)

  # mapper sim native doesn't use template free interface
  if(BASALT_INSTANTIATIONS_DOUBLE)
    add_executable(basalt_mapper_sim_naive src/mapper_sim_naive.cpp)
    target_link_libraries(basalt_mapper_sim_naive basalt_internal pangolin basalt::cli11)
  endif()

  add_executable(basalt_mapper src/mapper.cpp)
  target_link_libraries(basalt_mapper basalt_internal pangolin basalt::cli11)

  add_executable(basalt_opt_flow src/opt_flow.cpp)
  target_link_libraries(basalt_opt_flow basalt_internal pangolin basalt::cli11)

  add_executable(basalt_vio src/vio.cpp)
  target_link_libraries(basalt_vio basalt_internal pangolin basalt::cli11)

  add_executable(basalt_time_alignment src/time_alignment.cpp)
  target_link_libraries(basalt_time_alignment basalt_internal pangolin basalt::cli11)

  add_executable(basalt_kitti_eval src/kitti_eval.cpp)
  target_link_libraries(basalt_kitti_eval basalt::basalt-headers basalt::cli11)

  find_package(realsense2 QUIET)
  if(realsense2_FOUND)
    add_executable(basalt_rs_t265_record src/rs_t265_record.cpp src/device/rs_t265.cpp)
    target_link_libraries(basalt_rs_t265_record basalt_internal realsense2::realsense2 pangolin basalt::cli11)

    add_executable(basalt_rs_t265_vio src/rs_t265_vio.cpp src/device/rs_t265.cpp)
    target_link_libraries(basalt_rs_t265_vio basalt_internal realsense2::realsense2 pangolin basalt::cli11)
  endif()

endif(NOT BASALT_BUILD_SHARED_LIBRARY_ONLY)

# Update basalt.map library name (symbol present in the final .so)
configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake_modules/basalt.map.in ${CMAKE_CURRENT_BINARY_DIR}/basalt.map @ONLY)

# Export only the VIT interface symbols
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/basalt.map
             LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/basalt.map"
             VERSION "${VIT_VERSION_MAJOR}.${VIT_VERSION_MINOR}.${VIT_VERSION_PATCH}" SOVERSION ${VIT_VERSION_MAJOR}
)

if(BASALT_BUILD_SHARED_LIBRARY_ONLY)
  install(TARGETS basalt RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
          ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  )
  set(CONFIG_FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/msd/msdmo_config.json"
                   "${CMAKE_CURRENT_SOURCE_DIR}/data/msd/msdmo_calib.json"
  )
  set(UNIFIED_CONFIG_FILES msdmo.toml)

else()

  install(
    TARGETS basalt
            basalt_calibrate
            basalt_calibrate_imu
            basalt_vio_sim
            basalt_mapper_sim
            basalt_mapper
            basalt_opt_flow
            basalt_vio
            basalt_kitti_eval
            basalt_time_alignment
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  )

  if(BASALT_INSTANTIATIONS_DOUBLE)
    install(TARGETS basalt_mapper_sim_naive RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )
  endif()

  if(realsense2_FOUND)
    install(TARGETS basalt_rs_t265_record basalt_rs_t265_vio RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )
  endif()

  file(GLOB SCRIPTS_TO_INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/scripts/basalt_*.py")
  install(PROGRAMS ${SCRIPTS_TO_INSTALL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

  file(GLOB_RECURSE CONFIG_FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/*.json")
  file(GLOB ABSOLUTE_UNIFIED_CONFIG_FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/vit/*.toml.in")
  foreach(full_path IN LISTS ABSOLUTE_UNIFIED_CONFIG_FILES)
    get_filename_component(file_name "${full_path}" NAME_WE)
    list(APPEND UNIFIED_CONFIG_FILES ${file_name}.toml)
  endforeach()

endif(BASALT_BUILD_SHARED_LIBRARY_ONLY)

# Configure and install unified config toml files
set(CONFIG_FILE_INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/share/basalt) # Used for unified_config.in
foreach(unified_config IN LISTS UNIFIED_CONFIG_FILES)
  configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/data/vit/${unified_config}.in ${CMAKE_CURRENT_BINARY_DIR}/data/vit/${unified_config}
    @ONLY
  )
  list(APPEND CONFIG_FILES ${CMAKE_CURRENT_BINARY_DIR}/data/vit/${unified_config})
endforeach()

install(FILES ${CONFIG_FILES} DESTINATION ${CONFIG_FILE_INSTALL_DESTINATION})

configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake_modules/basalt.pc.in ${CMAKE_CURRENT_BINARY_DIR}/basalt.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/basalt.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)

# Replace install() to do-nothing macro.
macro(install)

endmacro()
if(BUILD_TESTS)
  # Include subproject (or any other CMake code) with "disabled" install().
  enable_testing()
  add_subdirectory(thirdparty/basalt-headers/test)
  add_subdirectory(test)
endif()
# Restore original install() behavior.
macro(install)
  _install(${ARGN})
endmacro()
