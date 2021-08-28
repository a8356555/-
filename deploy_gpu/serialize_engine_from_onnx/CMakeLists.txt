cmake_minimum_required(VERSION 3.12)
project(trt_serialize)
set(TENSORRT_LIB_PATH "/content/TensorRT-8.0.1.6/lib")
file(GLOB TensorRT_LIBS "${TENSORRT_LIB_PATH}/*.so")

add_executable(trt_serialize trt_parse_onnx_N_save.cpp)

include_directories(/content/TensorRT-8.0.1.6/include)
link_directories(/content/TensorRT-8.0.1.6/lib)
target_include_directories(trt_serialize PUBLIC /content/TensorRT-8.0.1.6/include)
target_link_libraries(trt_serialize PUBLIC ${TensorRT_LIBS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS   "-g")                     # 调试信息
set(CMAKE_CXX_FLAGS   "-Wall")                  # 开启所有警告