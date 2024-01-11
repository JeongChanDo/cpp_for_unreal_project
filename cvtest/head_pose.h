#pragma once
#include <opencv2/opencv.hpp>

const std::map<std::string, int> str2backend{
    {"opencv", cv::dnn::DNN_BACKEND_OPENCV}, {"cuda", cv::dnn::DNN_BACKEND_CUDA}
};
const std::map<std::string, int> str2target{
    {"cpu", cv::dnn::DNN_TARGET_CPU}, {"cuda", cv::dnn::DNN_TARGET_CUDA}, {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
};


class YuNet
{
public:
    YuNet(const std::string& model_path,
        const cv::Size& input_size,
        float conf_threshold,
        float nms_threshold,
        int top_k,
        int backend_id,
        int target_id);
    void setBackendAndTarget(int backend_id, int target_id);
    void setInputSize(const cv::Size& input_size);

    cv::Mat infer(const cv::Mat image);

private:
    cv::Ptr<cv::FaceDetectorYN> model;

    std::string model_path_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    int backend_id_;
    int target_id_;
};

cv::Mat get_image_points_2D();
cv::Mat get_figure_points_3D();
cv::Mat get_camera_matrix();
cv::Mat get_distortion_coeff();
void estimate_chin(cv::Mat& image_points_2D);
bool visualize(cv::Mat& image, const cv::Mat& faces, cv::Mat& image_points_2D, double fps);
std::vector<cv::Point2d> get_pose_points(cv::Mat& image_points_2D, cv::Mat& vector_rotation, cv::Mat& vector_translation, cv::Mat& camera_matrix, cv::Mat& distortion_coeff);
double clip_avg(std::vector<int> xdiff_vector);
