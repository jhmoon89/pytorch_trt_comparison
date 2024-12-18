cmake_minimum_required(VERSION 3.10)
project(trt_model_test)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include 디렉토리 설정
include_directories(
    ${PROJECT_SOURCE_DIR}/include  # 헤더 파일 경로
    ${OpenCV_INCLUDE_DIRS}         # OpenCV 헤더 파일 경로
    /usr/local/cuda/include        # CUDA 헤더 파일 경로
    /usr/include/x86_64-linux-gnu  # 시스템 경로
    /home/jihoon-epitone/Downloads/install_lenovo_part2-001/librealsense/include  # RealSense 경로
)

# 라이브러리 경로 설정
link_directories(
    /usr/local/lib
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
)

# TensorRTModel 모듈 추가 (inference.cpp)
add_library(inference
    src/inference.cpp
)

# TensorRTModel에 include 디렉토리 추가
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/include)

# TensorRT 라이브러리 연결
target_link_libraries(inference
    cudart
    nvinfer
    nvinfer_plugin
    nvonnxparser
)

# 메인 실행 파일 설정
add_executable(trt_model_test src/main.cpp)

# 메인 파일에 TensorRTModel 및 필요한 라이브러리 연결
target_link_libraries(trt_model_test
    inference       # inference 라이브러리 추가
    ${OpenCV_LIBS}  # OpenCV 라이브러리
    realsense2
)

# 디버그 모드 빌드 설정
set(CMAKE_BUILD_TYPE Debug)

# 컴파일러 옵션
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always -g")
