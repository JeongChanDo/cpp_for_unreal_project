#pragma once
#include <opencv2/opencv.hpp>
#include <vector>


class CV2NanoDetONNX {
    int STRIDES[3] = { 8, 16, 32 };
    int REG_MAX = 7;
    int data[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    cv::Mat PROJECT;

    cv::Vec3f mean = cv::Vec3f(103.53, 116.28, 123.675);
    cv::Vec3f std = cv::Vec3f(57.375, 57.12, 58.395);

    /*
    const float MEAN[3] = { 103.53f, 116.28f, 123.675f };
    const float STD[3] = { 57.375f, 57.12f, 58.395f };
    */
    cv::Size input_shape;
    float class_score_th;
    float nms_th;
    cv::dnn::Net net;
    std::vector<std::string> output_names;
    std::vector<cv::Mat> grid_points;

public:
    CV2NanoDetONNX(
        std::string model_path,
        int input_shape,
        float class_score_th,
        float nms_th
    );

    cv::Mat inference(cv::Mat image);
    cv::Mat make_grid_point(cv::Size grid_size, int stride);

    void resize_image(cv::Mat image, bool keep_ratio, cv::Mat& resized_image, int& new_height, int& new_width, int& top, int& left);

    cv::Mat preProcess(cv::Mat image);
    void postProcess(const std::vector<cv::Mat>& predict_results, std::vector<cv::Mat>& class_scores, std::vector<cv::Mat>& bbox_predicts, std::vector<cv::Mat>& class_ids);
    void get_bboxes_single(
        const std::vector<cv::Mat>& class_scores,
        const std::vector<cv::Mat>& bbox_predicts,
        float scale_factor,
        bool rescale = false,
        int topk = 1000
    );
    cv::Mat softmax(cv::Mat x, int axis = 1);


    std::vector<cv::Rect> getColorFilteredBoxes(cv::Mat image);
    cv::Mat draw_debug_roi(cv::Mat image, std::vector<cv::Rect> bboxes, std::vector<float> scores, std::vector<int> class_ids, int x, int y);
    
    
    

};
