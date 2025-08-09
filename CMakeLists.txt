cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(cmake-common.cmake)

add_executable(implicit src/implicit/ImplicitMain.cpp src/implicit/ImplicitSolver.cpp)
add_executable(mixed src/mixed/MixedMain.cpp src/mixed/MixedSolver.cpp)
add_executable(explicit src/explicit/ExplicitMain.cpp src/explicit/ExplicitSolver.cpp)

deal_ii_setup_target(implicit)
deal_ii_setup_target(mixed)
deal_ii_setup_target(explicit)