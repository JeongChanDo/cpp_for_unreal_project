#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

void drawRect(cv::Mat& frame, cv::Mat regressor, cv::Mat classificator, int stride, int anchor_count, int column, int row, int anchor, int offset) {
    int index = (int(row * 128 / stride) + column) * anchor_count + anchor + offset;
    float origin_score = regressor.at<float>(0, index, 0);
    float score = sigmoid(origin_score);
    if (score < 0.5) return;


    float x = classificator.at<float>(0, index, 0);
    float y = classificator.at<float>(0, index, 1);
    float w = classificator.at<float>(0, index, 2);
    float h = classificator.at<float>(0, index, 3);


    x += (column + 0.5) * stride - w / 2;
    y += (row + 0.5) * stride - h / 2;
    int ix = int(x);
    int iy = int(y);
    int iw = int(w);
    int ih = int(h);
    cv::rectangle(frame, cv::Rect(ix, iy, iw, ih), cv::Scalar(255, 0, 0), 1);
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Failed to open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::dnn::Net net = cv::dnn::readNetFromONNX("c:/palm_detection.onnx");
    std::vector<cv::String> outNames(2);
    outNames[0] = "regressors";
    outNames[1] = "classificators";


    while (true) {
        auto time_start = std::chrono::steady_clock::now();
        cap.read(frame);

        cv::resize(frame, frame, cv::Size(128, 128));
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);


        cv::Mat tensor;
        frame.convertTo(tensor, CV_32F, 1 / 127.5, -1.0);
        cv::Mat blob = cv::dnn::blobFromImage(tensor, 1.0, tensor.size(), 0, false, false, CV_32F);
        //std::cout << tensor.row(0) << std::endl << std::endl;
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, outNames);

        cv::Mat classificator = outputs[0];
        cv::Mat regressor = outputs[1];

        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 16; ++x) {
                for (int a = 0; a < 2; ++a) {
                    drawRect(frame, regressor, classificator, 8, 2, x, y, a, 0);
                }
            }
        }

        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                for (int a = 0; a < 6; ++a) {
                    drawRect(frame, regressor, classificator, 16, 6, x, y, a, 512);
                }
            }
        }

        cv::resize(frame, frame, cv::Size(640, 480));


        auto time_end = std::chrono::steady_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        std::string time_spent = "Time spent: " + std::to_string(time_diff) + "ms";
        cv::putText(frame, time_spent, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(125, 125, 125), 2);
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

        cv::imshow("Camera Streaming", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
