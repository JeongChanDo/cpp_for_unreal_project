
#include "NanoDet.h"


int main() {
    /*
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Failed to open camera" << std::endl;
        return -1;
    }
    */
    cv::Mat frame;


    CV2NanoDetONNX net = CV2NanoDetONNX::CV2NanoDetONNX(
        "nanodet_finger_v3_sim.onnx",
        320,
        0.35f,
        0.6f
    );

    //double fps = cap.get(cv::CAP_PROP_FPS);
    //double delay = 1000 / fps;
    //double elapsed_time = 0;


    
    frame = cv::imread("test.jpg");

    cv::resize(frame, frame, cv::Size(640, 480));


    net.inference(frame);

    //imshow("test", frame);

    //destroyAllWindows();


    /*
    while (true) {
        bool ret = cap.read(frame);
        if (!ret)
        {
            break;
        }

        double start_time = cv::getTickCount();
        if (elapsed_time > delay)
        {
            elapsed_time = elapsed_time - delay;
            if (elapsed_time <= 0)
            {
                // Do nothing
            }
            else
                continue;
        }

        cv::Mat debug_image = frame.clone();
        cv::Mat skin_image;


        */

        /*

        cv::Mat tensor;
        frame.convertTo(tensor, CV_32F, 1 / 127.5, -1.0);
        cv::Mat blob = cv::dnn::blobFromImage(tensor, 1.0, tensor.size(), 0, false, false, CV_32F);
        //std::cout << tensor.row(0) << std::endl << std::endl;
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, outNames);

        cv::Mat classificator = outputs[0];
        cv::Mat regressor = outputs[1];

        cv::resize(frame, frame, cv::Size(640, 480));
        */
    /*
        auto time_end = std::chrono::steady_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        std::string time_spent = "Time spent: " + std::to_string(time_diff) + "ms";
        
        //cv::putText(frame, time_spent, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(125, 125, 125), 2);
        //cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

        cv::imshow("Camera Streaming", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    */
    return 0;
}
