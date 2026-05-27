# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# external/cudss/CMakeLists.txt
#
# Plugin CMakeLists for the cuDSS coarse-grid solver.
# Included by the top-level CMakeLists.txt when AMGX_USE_CUDSS=ON.
#
# This file:
#   1. Finds the cuDSS package (errors out if not found)
#   2. Adds cudss_solver.cu to the amgx_libs OBJECT library
#   3. Wires include directories and link libraries for amgx / amgxsh
#   4. Optionally builds the unit test and benchmark

cmake_minimum_required(VERSION 3.18)

# -----------------------------------------------------------------------
# 1. Find cuDSS
# -----------------------------------------------------------------------
find_package(cudss REQUIRED
    HINTS ${CUDSS_ROOT} $ENV{CUDSS_ROOT}
          /usr/local/cudss /opt/nvidia/cudss)

if(NOT cudss_FOUND)
    message(FATAL_ERROR
        "cuDSS not found. Set CUDSS_ROOT or AMGX_USE_CUDSS=OFF.\n"
        "Download cuDSS from https://developer.nvidia.com/cudss")
endif()

message(STATUS "cuDSS plugin: found cuDSS at ${cudss_INCLUDE_DIRS}")

# -----------------------------------------------------------------------
# 2. Add the real implementation to amgx_libs
# -----------------------------------------------------------------------
target_sources(amgx_libs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cudss_solver.cu)

# The plugin header lives in this directory; expose it so cudss_solver.cu
# can #include "cudss_solver.h" and so that amgx_libs consumers can reach it.
target_include_directories(amgx_libs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${cudss_INCLUDE_DIRS})

# -----------------------------------------------------------------------
# 3. Link cuDSS into the amgx static and shared libraries
# -----------------------------------------------------------------------
target_link_libraries(amgx   CUDA::cudss)
target_link_libraries(amgxsh CUDA::cudss)

# -----------------------------------------------------------------------
# 4. Optional: unit test (compiled only when tests are enabled)
# -----------------------------------------------------------------------
# The test is compiled into the existing amgx_tests_launcher target.
# We add the source here so it is picked up when AMGX_USE_CUDSS=ON.
if(TARGET amgx_tests_libs)
    target_sources(amgx_tests_libs PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cudss_solver_test.cu)
    target_include_directories(amgx_tests_libs PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${cudss_INCLUDE_DIRS})
endif()

# -----------------------------------------------------------------------
# 5. Optional: benchmark executable
# -----------------------------------------------------------------------
if(TARGET amgxsh)
    add_executable(bench_cudss_vs_dense_lu
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_cudss_vs_dense_lu.c)

    target_include_directories(bench_cudss_vs_dense_lu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    GET_FILENAME_COMPONENT(CMAKE_C_COMPILER_NAME "${CMAKE_C_COMPILER}" NAME)
    IF(CMAKE_C_COMPILER_NAME MATCHES cl AND NOT CMAKE_C_COMPILER_NAME MATCHES clang)
        set(bench_dyn_libs amgxsh CUDA::cudart_static CUDA::cublas)
    ELSE()
        set(bench_dyn_libs amgxsh CUDA::cudart_static CUDA::cublas rt dl)
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/bench_cudss_vs_dense_lu.c
            PROPERTIES COMPILE_FLAGS -std=c99)
    ENDIF()

    target_link_libraries(bench_cudss_vs_dense_lu ${bench_dyn_libs})
endif()
