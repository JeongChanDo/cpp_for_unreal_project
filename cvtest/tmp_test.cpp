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

    std::cout << index << std::endl;
    std::cout << score << std::endl << std::endl;

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

void printMat(const cv::Mat& mat) {
    for (int i = 0; i < mat.size().height; ++i) {
        for (int j = 0; j < mat.size().width; ++j) {
            std::cout << mat.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

int main() {

    cv::Mat frame = cv::imread("frame.jpg");
    cv::dnn::Net net = cv::dnn::readNetFromONNX("c:/palm_detection.onnx");
    auto names = net.getUnconnectedOutLayersNames();
    std::vector<cv::String> outNames(2);
    outNames[0] = "regressors";
    outNames[1] = "classificators";


    for (int i = 0; i < names.size(); i++)
    {
        std::cout << names[i] << typeid(names[i]).name() << std::endl;
    }


    cv::resize(frame, frame, cv::Size(128, 128));
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    //std::cout << frame.row(100) << std::endl << std::endl;


    cv::Mat tensor;
    frame.convertTo(tensor, CV_32F, 1 / 127.5, -1.0);
    cv::Mat blob = cv::dnn::blobFromImage(tensor, 1.0, tensor.size(), 0, false, false, CV_32F);
    //std::cout << tensor.row(0) << std::endl << std::endl;
    net.setInput(blob);
  
    std::vector<cv::Mat> outputs;

    /*
    auto tmp = net.forward("regressors");
    std::cout << tmp.size() << std::endl;
    */


    net.forward(outputs, outNames);

    cv::Mat classificator = outputs[0];
    cv::Mat regressor = outputs[1];


    std::cout << outputs[0].rows << std::endl;
    std::cout << outputs[1].rows << std::endl;

    std::cout << classificator.size[0] << std::endl;
    std::cout << classificator.size[1] << std::endl;
    std::cout << classificator.size[2] << std::endl;

    std::cout << regressor.size[0] << std::endl;
    std::cout << regressor.size[1] << std::endl;
    std::cout << regressor.size[2] << std::endl;

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
    /*
    for (int i = 0; i < 50; i++) {
        std::cout << regressor.at<float>(0, i, 0) << std::endl;
    }

    for (int i = 0; i < 50; i++) {
        std::cout << classificator.at<float>(0, i, 0) << classificator.at<float>(0, i, 1) << classificator.at<float>(0, i, 2)  << std::endl;
    }
    */



    /*
    cv::Mat regressor = outputs[0];
    cv::Mat classificator = outputs[1];
    std::cout << "print mat " << std::endl;
    std::cout << "rows : " << regressor.rows << ",cols : " << regressor.cols << std::endl;
    printMat(regressor);
    */
    /*
    std::cout << regressor.size() << std::endl;
    std::cout << regressor.size().width << std::endl;
    std::cout << regressor.size().height << std::endl;

    std::cout << classificator.size() << std::endl;
    std::cout << classificator.size().width << std::endl;
    std::cout << classificator.size().height << std::endl;
    */

    /*
    float* regressor = (float*)outputs[0].data;
    float* classificator = (float*)outputs[1].data;

    std::cout << outputs.size() << std::endl;
    std::cout << outputs[0].size() << std::endl;
    std::cout << outputs[0].elemSize() << std::endl;
    std::cout << outputs[1].size() << std::endl;
    std::cout << outputs[1].elemSize() << std::endl;

    cv::Size outputs_size = outputs[0].size();
    int num_channels = outputs[0].channels();
    int blob_width = outputs_size.width;
    int blob_height = outputs_size.height;

    std::cout << num_channels << " " << blob_width << " " << blob_height << " " << std::endl;
    outputs_size = outputs[1].size();
    num_channels = outputs[1].channels();
    blob_width = outputs_size.width;
    blob_height = outputs_size.height;

    std::cout << num_channels << " " << blob_width << " " << blob_height << " " << std::endl;



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
    */
    cv::resize(frame, frame, cv::Size(640, 480));


    auto time_end = std::chrono::steady_clock::now();
    cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

    cv::imshow("Camera Streaming", frame);

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}
