cmake_minimum_required(VERSION 3.10)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

project(L-Shapefit)

include_directories(.)

add_executable(${PROJECT_NAME} lshaped_fit.hpp test_lshaped_fitting.cpp)
