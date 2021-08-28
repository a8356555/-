#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <memory>

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <inference.h>

using namespace cv;
using namespace cv::cuda;
using cv::cuda::GpuMat;
// trt logger

void Logger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
        if ((severity == Severity::kERROR) || severity == Severity::kINTERNAL_ERROR)
            std::cout<< msg << "\n";
}
Logger gLogger;

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}


std::vector<std::string> getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes name. \n";
        return classes;
    }
    
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}



void preProcessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims) {
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "Input image: " << image_path << " load failed\n";
        return;
    }    
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    int h = frame.rows;
    int w = frame.cols;
    int dh_half, dw_half;
    if (h > w) {
        dh_half = static_cast<int>(0.1*h/2);
        dw_half = static_cast<int>((h+2*dh_half-w)/2);
    } else {
        dw_half = static_cast<int>(0.1*w/2);
        dh_half = static_cast<int>((w+2*dw_half-h)/2);
    }
    GpuMat gpu_frame;
    gpu_frame.upload(frame);
    
    // copymakeborder
    cuda::copyMakeBorder(gpu_frame, gpu_frame, dh_half, dh_half, dw_half, dw_half, cv::BORDER_REPLICATE);

    // resize
    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    auto target_size = cv::Size(248, 248);
    cuda::resize(gpu_frame, gpu_frame, target_size);
    
    // crop
    cv::Rect ROI(12, 12, 236, 236);
    gpu_frame = gpu_frame(ROI).clone();

    // normalize
    /*gpu_frame.convertTo(gpu_frame, cv::CV_32FC3, 1.f/255.f);
    gpu_frame = cuda::subtract(gpu_frame, cv::Scalar(0.485f, 0.456f, 0.406f), gpu_frame);
    gpu_frame = cuda::divide(gpu_frame, cv::Scalar(0.229f, 0.224f, 0.225f), gpu_grame);*/
    
    std::vector<GpuMat> chw;

    // copy processed data to gpu_input channel by channel using pointer
    // 1) pass pointer into vector 2) split processed image into vector     
    for (size_t i=0; i<channels; ++i)
    {
        chw.emplace_back(GpuMat(input_size, CV_32FC1, gpu_input + i*input_width*input_height));
    }   
    cuda::split(gpu_frame, chw);
    return;
}


void postProcessResults(float* gpu_output, std::vector<float>& cpu_output, const nvinfer1::Dims& dims, int batch_size)
{
    // auto classes = getClassNames("imagenet_classes.txt");
    int class_num = 801;
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
    // calculate softmax
    /* std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val){return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0);
 
    // calculate indices and sort
    std::vector<int> indices((getSizeByDim(dims)*batch_size));
    std::iota(indices.begin(), indices.end(), 0);
    // order prediction indices by its value
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2){ return cpu_output[i1]>cpu_output[i2];});
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.5)
    {
        if ( indices[i] < classes.size() )
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
     
        if (indices[i] < class_num)
            std::cout << "pred" << indices[i] << " | ";
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "\n";
        ++i;
    }*/
    return;
}

void loadEngine(std::string const& path, nvinfer1::ICudaEngine*& engine)
{
    std::ifstream engineFile(path, std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Error opening engine file: " << engine << std::endl;
        return; 
    }
    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cerr << "Error loading engine file: " << engine << std::endl;
        return;
    }
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    return;
}


void buildTRTEngineContextBuffer(
    int batch_size,
    const std::string& engine_path, 
    nvinfer1::ICudaEngine*& engine, 
    nvinfer1::IExecutionContext*& context, 
    std::vector<nvinfer1::Dims>& input_dims,
    std::vector<nvinfer1::Dims>& output_dims,
    std::vector<void*>& buffers ) 
{
    loadEngine(engine_path, engine);
    context = engine->createExecutionContext();
    buffers = std::vector<void*>(engine->getNbBindings());

    for (size_t i=0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return;
    }
    return;
}

std::vector<float> predict(const std::string& engine_path, const std::string& image_path) 
{
    int batch_size = 1;
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};
    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;
    std::vector<void*> buffers;
    buildTRTEngineContextBuffer(batch_size, engine_path, engine, context, input_dims, output_dims, buffers);

    preProcessImage(image_path, (float*)buffers[0], input_dims[0]);
    context->enqueue(batch_size, buffers.data(), 0, nullptr);
    
    std::vector<float> cpu_output((getSizeByDim(output_dims[0])*batch_size));        
    postProcessResults((float*)buffers[1], cpu_output, output_dims[0], batch_size);

    for (void* buf:buffers)
    {
        cudaFree(buf);
    }
    return cpu_output;
}

void evaluate_predict_speed(const std::string& engine_path, const std::string& image_path, int test_num)
{
    int batch_size = 1;
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};
    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;
    std::vector<void*> buffers;
    buildTRTEngineContextBuffer(batch_size, engine_path, engine, context, input_dims, output_dims, buffers);
    
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<test_num; i++) {
        std::vector<float> cpu_output((getSizeByDim(output_dims[0])*batch_size));
        preProcessImage(image_path, (float*)buffers[0], input_dims[0]);
        context->enqueue(batch_size, buffers.data(), 0, nullptr);
        
        postProcessResults((float*)buffers[1], cpu_output, output_dims[0], batch_size);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    auto elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count()/test_num;
    std::cout << elapsed_time_ms/1000 << std::endl;

    for (void* buf:buffers)
    {
        cudaFree(buf);
    }
    return;
}