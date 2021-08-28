#include <string>
#include "trt_inference.h"

int main(int argc, char* argv[])
{
    int n;
    if (argc<3)
    {
        std::cerr << "basic usage: " << argv[0] << " engine.trt image.jpg\n" << "test speed usage: " << argv[0] << " engine.trt image.jpg " << std::endl;
        
        return -1;
    }
    n = static_cast<int>(argv[3]);
    std::string engine_path(argv[1]);
    std::string image_path(argv[2]);

    
    predict()
    return 0;
}