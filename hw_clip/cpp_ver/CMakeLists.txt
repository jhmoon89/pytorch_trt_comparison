cmake_minimum_required(VERSION 3.10)
project(trt_model_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories setup
include_directories(
    ${PROJECT_SOURCE_DIR}/include  # Path to header files
    ${OpenCV_INCLUDE_DIRS}         # Path to OpenCV header files
    /usr/local/cuda/include        # Path to CUDA header files
    /usr/include/x86_64-linux-gnu  # System path
    #/home/jihoon-epitone/Downloads/install_lenovo_part2-001/librealsense/include  # Path to RealSense (commented out)
)

# Library directories setup
link_directories(
    /usr/local/lib
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
)

# Add TensorRTModel module (inference.cpp)
add_library(inference
    src/inference.cpp
)

# Add include directories for TensorRTModel
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Link TensorRT libraries
target_link_libraries(inference
    cudart
    nvinfer
    nvinfer_plugin
    nvonnxparser
)

# Setup main executable
add_executable(trt_model_test src/main.cpp)

# Link TensorRTModel and required libraries to main executable
target_link_libraries(trt_model_test
    inference       # Add inference library
    ${OpenCV_LIBS}  # Link OpenCV libraries
    #realsense2     # (commented out) Link RealSense library
)

# Set build type to Debug
set(CMAKE_BUILD_TYPE Debug)

# Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always -g")
