cmake_minimum_required(VERSION 3.15)
project(slam_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The FP contract (benchmarks/CPP_BACKEND_MIGRATION.md, locked decision
# 2): scalar IEEE-754 only - FMA contraction disabled and no fast-math,
# so kernels written in the Python-side operation order are
# bit-identical to numpy across machines. Do NOT add optimization flags
# that relax floating-point semantics.
if(MSVC)
  add_compile_options(/fp:precise)
else()
  add_compile_options(-ffp-contract=off -fno-fast-math)
endif()

find_package(pybind11 CONFIG REQUIRED)

# The vendored g2o (the exact source g2o-python 0.0.12 builds from,
# library version 1.1.0) + the vendored Eigen it needs. Static libs
# bundled into our extension; only the Eigen linear solver (no
# CHOLMOD/CSPARSE/OpenGL/logging). g2o's internals are the sanctioned
# FP exception in the locked decision 2 contract.
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
set(BUILD_SHARED_LIBS OFF)
set(G2O_USE_CHOLMOD OFF)
set(G2O_USE_CSPARSE OFF)
set(G2O_USE_OPENGL OFF)
set(G2O_USE_LOGGING OFF)
set(G2O_BUILD_APPS OFF)
set(G2O_BUILD_EXAMPLES OFF)
set(G2O_BUILD_BENCHMARKS OFF)
set(G2O_BUILD_UNITTESTS OFF)
set(G2O_BUILD_TESTS OFF)
add_subdirectory(third_party/g2o third_party/g2o-build EXCLUDE_FROM_ALL)

pybind11_add_module(slam_native
                    src/slam_native.cpp
                    src/map_core.cpp
                    src/error_fold.cpp
                    src/ba_solver.cpp)
target_link_libraries(slam_native PRIVATE
                      core types_sba types_slam3d solver_eigen stuff)

# The module installs at the site-packages root (importable as
# `slam_native`): the extension IS the Python/C++ boundary.
install(TARGETS slam_native DESTINATION .)
