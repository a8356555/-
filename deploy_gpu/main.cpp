#include <string>
#include <vector>
#include "trt_inference.h"

int main(int argc, char* argv[])
{
    int n;
    if (argc<3)
    {
        std::cerr << "basic usage: " << argv[0] << " engine.trt image.jpg\n" << "test speed usage: " << argv[0] << " engine.trt image.jpg test_loop_number" << std::endl;        
        return -1;
    } else if (argc == 3){
        std::string engine_path(argv[1]);
        std::string image_path(argv[2]);                
        std::vector<float> cpu_result;
        cpu_result = predict(engine_path, image_path);
    } else {
        std::string engine_path(argv[1]);
        std::string image_path(argv[2]);        
        int test_num = static_cast<int>(argv[3]);
        evaluate_predict_speed(engine_path, image_path, test_num);
    }    
    return 0;
}