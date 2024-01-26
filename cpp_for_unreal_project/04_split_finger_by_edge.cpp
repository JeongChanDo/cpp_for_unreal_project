#include "NanoDet.h"

int main()
{
    dnn::Net tmp = dnn::readNetFromONNX("nanodet_finger_v3_sim.onnx");

    NanoDet nanodet(320, 0.35, 0.4);

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

    Rect roi1, roi2;
    Mat roi_hand1, roi_hand2;
    Mat roi_hand_inv1, roi_hand_inv2;

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
        Mat skin_image, edges, canny;
        vector<Rect> color_boxes = nanodet.get_color_filtered_boxes(frame, skin_image);

        cv::cvtColor(skin_image, edges, cv::COLOR_BGR2GRAY); // 그레이스케일로 변환
        // 가우시안 스무딩 적용
        cv::GaussianBlur(edges, edges, cv::Size(3, 3), 0);

        cv::Canny(edges, canny, 10, 50); // 에지 추출

        if (color_boxes.size() > 0)
        {
            roi1 = color_boxes[0];
            roi1.height = roi1.height / 5;
            roi_hand1 = canny(roi1).clone();
            cv::bitwise_not(roi_hand1, roi_hand_inv1);



            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Rect> bounding_boxes;
            std::vector<cv::Rect> sorted_bounding_boxes;

            cv::findContours(roi_hand1, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                cv::Rect bbox = cv::boundingRect(contour);
                cv::rectangle(roi_hand1, Point(bbox.x, bbox.y), Point(bbox.x + bbox.width, bbox.y + bbox.height), Scalar(255), 1);
                if ((bbox.width * bbox.height > 10 * 10) || (bbox.width * bbox.height < 100 * 100)) {
                    bounding_boxes.push_back(bbox);
                }
            }

            for (const auto& bbox : bounding_boxes) {
                cv::rectangle(skin_image, Point(roi1.x + bbox.x, roi1.y + bbox.y), Point(roi1.x + bbox.x + bbox.width, roi1.y + bbox.y + bbox.height), Scalar(255, 0, 0), 2);
            }

            /*
            sort(bounding_boxes.begin(), bounding_boxes.end(), nanodet.compareRectByArea);

            for (int i = 0; i < 3; i++)
            {
                if (i == 3) break;
                cv::Rect bbox = bounding_boxes[i];
                //cv::rectangle(roi_hand1, Point(bbox.x, bbox.y), Point(bbox.x + bbox.width, bbox.y + bbox.height), Scalar(255), 1);
                cv::rectangle(skin_image, Point(roi1.x + bbox.x, roi1.y + bbox.y), Point(roi1.x + bbox.x + bbox.width, roi1.y + bbox.y + bbox.height), Scalar(255, 0, 0), 2);

                sorted_bounding_boxes.push_back(bbox);
            }
            */
            cv::imshow("hand1", roi_hand1);
            cv::imshow("hand_inv1", roi_hand_inv1);

        }

        if (color_boxes.size() == 2)
        {

            roi2 = color_boxes[1];
            roi2.height = roi2.height / 5;
            roi_hand2 = canny(roi2).clone();
            cv::bitwise_not(roi_hand2, roi_hand_inv2);





            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Rect> bounding_boxes;
            std::vector<cv::Rect> sorted_bounding_boxes;

            cv::findContours(roi_hand2, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                cv::Rect bbox = cv::boundingRect(contour);
                cv::rectangle(roi_hand2, Point(bbox.x, bbox.y), Point(bbox.x + bbox.width, bbox.y + bbox.height), Scalar(255), 1);
                if ((bbox.width * bbox.height > 10 * 10) || (bbox.width * bbox.height < 100 * 100)) {
                    bounding_boxes.push_back(bbox);
                }
            }

            for (const auto& bbox : bounding_boxes) {
                cv::rectangle(skin_image, Point(roi2.x + bbox.x, roi2.y + bbox.y), Point(roi2.x + bbox.x + bbox.width, roi2.y + bbox.y + bbox.height), Scalar(255, 0, 0), 2);
            }





            cv::imshow("hand2", roi_hand2);
            cv::imshow("hand_inv2", roi_hand_inv2);
        }

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
        //nanonet.detect(skin_image);
        double end_tick = cv::getTickCount();
        elapsed_time = floor((end_tick - start_tick) * 1000 / cv::getTickFrequency());

        std::stringstream ss;
        ss << "Elapsed Time : " << elapsed_time << " ms";
        cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Camera Streaming", frame);
        cv::imshow("edges", edges);
        cv::imshow("canny", canny);
        cv::imshow("skin_image", skin_image);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();



}