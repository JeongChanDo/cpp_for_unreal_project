#include <opencv2/opencv.hpp>

using namespace std;
int main() {
    cv::Mat img = cv::Mat(21, 3, CV_32FC1);

    cout << img.size() << endl;
    cout << img.size[0] <<","<<img.size[1] << endl;

    cv::Mat img2;

    cout << img2.size() << endl;
    cout << img2.size[0] << "," << img2.size[1] << endl;

    return 0;
}
