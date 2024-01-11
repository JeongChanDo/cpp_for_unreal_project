
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{

    double dA[] = {
        1, 2
    };
    cv::Mat A = cv::Mat(1, 2, CV_32FC1, dA);

    double dB[] = {
        1, 2
    };
    cv::Mat B = cv::Mat(1, 2, CV_32FC1, dB);
    std::cout << "A rows : " << A.rows << ", A cols : " << A.cols << std::endl;
    std::cout << "B rows : " << B.rows << ", B cols : " << B.cols << std::endl;

    double b = A.dot(B);
    std::cout << b<< std::endl;


}
