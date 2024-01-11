
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    //https://answers.opencv.org/question/170221/cvsortidx-opencv-32/
    /*
    cv::Mat unsorted(1, 5, CV_32F);
    unsorted.at<int>(0, 0) = 40;
    unsorted.at<int>(0, 1) = 30;
    unsorted.at<int>(0, 2) = 100;
    unsorted.at<int>(0, 3) = 110;
    unsorted.at<int>(0, 4) = 10;

    cv::Mat sorted;
    cv::sortIdx(unsorted, sorted,  cv::SORT_DESCENDING);

    std::cout << sorted.at<int>(0, 0) << " " << sorted.at<int>(0, 1) << " " << sorted.at<int>(0, 2) << " " << sorted.at<int>(0, 3) << " " << sorted.at<int>(0, 4) << " " << std::endl;
    */

    //https://cppsecrets.com/users/20251211111031011151049910497110100114971079711011610455565764103109971051084699111109/C00-OpenCVsortIdx.php
    float x;
    cv::Mat A = cv::Mat::eye(3, 3, CV_32F);
    std::cout << "Enter the elements of 3*3 Matrix\n";
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cin >> x;
            A.at<float>(i, j) = x;
        }
    }
    cv::Mat B = cv::Mat::zeros(3, 3, CV_32F);
    cv::sortIdx(A, B, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
    std::cout << "Sorted Matrix is\n";
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            printf("%d	", B.at<int>(i, j));
        printf("\n");
    }
    return 0;


}
