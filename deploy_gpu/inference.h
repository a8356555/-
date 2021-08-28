#pragma once
#include <NvInfer.h>
#include <vector>
#include <string>

class Logger: public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
};

size_t getSizeByDim(const nvinfer1::Dims& dims);

std::vector<std::string> getClassNames(const std::string& imagenet_class);

void preProcessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims);

void postProcessResults(float* gpu_output, std::vector<float>& cpu_output, const nvinfer1::Dims& dims, int batch_size);

void loadEngine(std::string const& path, nvinfer1::ICudaEngine*& engine);

std::vector<float> predict(const std::string& engine_path, const std::string& image_path);

void evaluate_predict_speed(const std::string& engine_path, const std::string& image_path, int test_num);
