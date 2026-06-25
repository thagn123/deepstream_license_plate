#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

int main() {
    int count = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << count << std::endl;
    return 0;
}
