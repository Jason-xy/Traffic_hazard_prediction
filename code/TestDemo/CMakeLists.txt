  
cmake_minimum_required(VERSION 2.8)

project(CarDetection)

find_package(jetson-utils)
find_package(jetson-inference)
find_package(CUDA)
find_package(OpenCV REQUIRED)
find_package(Qt4)

include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

cuda_add_executable(CarDetection carDetection.cpp)

target_link_libraries(CarDetection jetson-inference)