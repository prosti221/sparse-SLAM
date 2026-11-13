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

pybind11_add_module(slam_native src/slam_native.cpp)

# The module installs at the site-packages root (importable as
# `slam_native`): the extension IS the Python/C++ boundary.
install(TARGETS slam_native DESTINATION .)
