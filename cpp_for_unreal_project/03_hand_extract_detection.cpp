#include "NanoDet.h"

int main()
{
    dnn::Net tmp = dnn::readNetFromONNX("nanodet_finger_v3_sim.onnx");

    NanoDet nanonet(320, 0.35, 0.4);

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
        Mat skin_image;
        vector<Rect> color_boxes = nanonet.get_color_filtered_boxes(frame, skin_image);
        /*
        cout << "box size : " <<  color_boxes.size() << endl;
        for (int i = 0; i < color_boxes.size(); i++)
        {
            cout << "skin_image : " << skin_image.size() << endl;
            cout << "idx : " << i << endl;
            Rect roi = color_boxes[i];
            cout << roi << endl;
            Mat roi_hand = skin_image(roi);
            nanonet.detect(roi_hand);

        }
        */
        nanonet.detect(skin_image);
        double end_tick = cv::getTickCount();
        elapsed_time = floor((end_tick - start_tick) * 1000 / cv::getTickFrequency());

        std::stringstream ss;
        ss << "Elapsed Time : " << elapsed_time << " ms";
        cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Camera Streaming", frame);
        cv::imshow("skin_image", skin_image);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();


    
}