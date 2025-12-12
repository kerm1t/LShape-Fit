cmake_minimum_required(VERSION 3.10)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

project(L-Shapefit)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} .)

add_executable(${PROJECT_NAME} lshaped_fit.cpp lshaped_fit.h test_lshaped_fitting.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

