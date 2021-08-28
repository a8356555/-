

int main(int argc, char* argv[])
{
    int n;
    if (argc<3)
    {
        std::cerr << "usage " << argv[0] << " engine.trt image.jpg\n";
        int n=100;
        return -1;
    }
    n = static_cast<int>(argv[3]);
    std::string engine_path(argv[1]);
    std::string image_path(argv[2]);

    
    predict()
    return 0;
}