#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int ac, char** av) {

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		printf("Can't open the camera");
		return -1;
	}

	Mat img;
	while (1)
	{
		cap >> img;
		Mat bgraImage;

		cv::cvtColor(img, bgraImage, cv::COLOR_BGR2BGRA);
		imshow("camera img", bgraImage);
		if (waitKey(1) == 27)
			break;
	}
	return 0;
}