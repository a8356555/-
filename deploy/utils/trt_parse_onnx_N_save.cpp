#include <iostream>
#include <fstream> 
#include <vector>
#include <string>
#include <NvInfer.h>
#include <NvOnnxParser.h>


class Logger: public nvinfer1::ILogger {
public:
void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if ((severity == Severity::kERROR) || severity == Severity::kINTERNAL_ERROR)
            std::cout<< msg << "\n";
    }
} gLogger;

void parseOnnxModel(
    const std::string& model_path, 
    nvinfer1::ICudaEngine*& engine, 
    nvinfer1::IExecutionContext*& context)
{
    nvinfer1::IBuilder* builder{nvinfer1::createInferBuilder(gLogger)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  
    nvinfer1::INetworkDefinition* network{builder->createNetworkV2(explicitBatch)};
    nvonnxparser::IParser* parser{nvonnxparser::createParser(*network, gLogger)};
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
	for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << parser->getError(i)->desc() << std::endl;
	}        

    nvinfer1::IBuilderConfig* config{builder->createBuilderConfig()};
    // 1GB memory 1byte*(2**10)*(2**10)*(2**10)
    config->setMaxWorkspaceSize(1ULL << 30);
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    builder->setMaxBatchSize(1);
    
    engine = builder->buildEngineWithConfig(*network, *config);
    context = engine->createExecutionContext();
    return;
}

int main(int argc, char* argv[])
{
    if (argc<3)
    {
        std::cerr << "usage " << argv[0] << " model_onnx_path(model.onnx) target_engine_path(engine.trt)\n";
        return -1;
    }
    
    
    std::string model_path(argv[1]);
    std::string target_path(argv[2]);
 
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};
    parseOnnxModel(model_path, engine, context);
    std::cout << "Parsing onnx model done" << std::endl;

    nvinfer1::IHostMemory *serializedModel = engine->serialize();
    std::ofstream ofs(target_path, std::ios::out | std::ios::binary);
    ofs.write((char*)(serializedModel->data()), serializedModel->size());
    ofs.close();
    std::cout << "Saving serialized engine done" << std::endl;
    return 0;
}