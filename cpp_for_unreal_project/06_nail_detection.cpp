#include "NanoDet.h"

int main()
{

    NanoDet nanodet(320, 0.4, 0.4);

    cv::Mat frame;

    string rtsp_url = "rtsp://192.168.0.181:8080/video/h264";
    //cv::VideoCapture cap(0);
    cv::VideoCapture cap(rtsp_url);
    if (!cap.isOpened()) {
        std::cout << "Failed to open camera" << std::endl;
        return -1;
    }


    double delay = 50;
    double elapsed_time = 0;

    while (true) {
        bool ret = cap.read(frame);
        if (!ret)
        {
            break;
        }

        double start_tick = cv::getTickCount();
        //cout << elapsed_time << "," << delay << endl;
        if (elapsed_time > delay)
        {
            elapsed_time = elapsed_time - delay;
            if (elapsed_time > 0)
            {
                continue;
            }

        }
        nanodet.detect(frame);

        double end_tick = cv::getTickCount();
        elapsed_time = floor((end_tick - start_tick) * 1000 / cv::getTickFrequency());

        std::stringstream ss;
        ss << "Elapsed Time : " << elapsed_time << " ms";
        cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Camera Streaming", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();



}