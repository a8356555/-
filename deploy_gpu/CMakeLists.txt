cmake_minimum_required(VERSION 3.12)
project(trt_inference)
set(TENSORRT_LIB_PATH "/content/TensorRT-8.0.1.6/lib")
file(GLOB TensorRT_LIBS "${TENSORRT_LIB_PATH}/*.so")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} /content/TensorRT-8.0.1.6/include ${PROJECT_SOURCE_DIR})
link_directories(/content/TensorRT-8.0.1.6/lib ${OpenCV_LIBS})

add_executable(trt_inference main.cpp inference.cpp)

target_include_directories(trt_inference PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} /content/TensorRT-8.0.1.6/include)
target_link_libraries(trt_inference PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS   "-g")                     # 调试信息
set(CMAKE_CXX_FLAGS   "-Wall")                  # 开启所有警告