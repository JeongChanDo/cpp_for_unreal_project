#include "head_pose.h"
#include <numeric>

YuNet::YuNet(const std::string& model_path,
    const cv::Size& input_size = cv::Size(320, 320),
    float conf_threshold = 0.6f,
    float nms_threshold = 0.3f,
    int top_k = 5000,
    int backend_id = 0,
    int target_id = 0)
    : model_path_(model_path), input_size_(input_size),
    conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
    top_k_(top_k), backend_id_(backend_id), target_id_(target_id)
{
    model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
}

void YuNet::setBackendAndTarget(int backend_id, int target_id)
{
    backend_id_ = backend_id;
    target_id_ = target_id;
    model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
}

void YuNet::setInputSize(const cv::Size& input_size)
{
    input_size_ = input_size;
    model->setInputSize(input_size_);
}

cv::Mat YuNet::infer(const cv::Mat image)
{
    cv::Mat res;
    model->detect(image, res);
    return res;
}


cv::Mat get_image_points_2D()
{
    cv::Mat image_points_2D = cv::Mat::zeros(6, 2, CV_64F);
    image_points_2D.at<double>(0, 0) = 0;  // right eye
    image_points_2D.at<double>(0, 1) = 0;
    image_points_2D.at<double>(1, 0) = 0;  // left eye
    image_points_2D.at<double>(1, 1) = 0;
    image_points_2D.at<double>(2, 0) = 0;  // nose tip
    image_points_2D.at<double>(2, 1) = 0;
    image_points_2D.at<double>(3, 0) = 0;  // right mouth corner
    image_points_2D.at<double>(3, 1) = 0;
    image_points_2D.at<double>(4, 0) = 0;  // left mouth corner
    image_points_2D.at<double>(4, 1) = 0;
    image_points_2D.at<double>(5, 0) = 0;  // chin
    image_points_2D.at<double>(5, 1) = 0;
    return image_points_2D;
}

cv::Mat get_figure_points_3D()
{
    cv::Mat figure_points_3D = cv::Mat::zeros(6, 3, CV_64F);
    figure_points_3D.at<double>(0, 0) = 180.0;     // Right eye right corner
    figure_points_3D.at<double>(0, 1) = 170.0;
    figure_points_3D.at<double>(0, 2) = -135.0;
    figure_points_3D.at<double>(1, 0) = -180.0;    // Left eye left corner
    figure_points_3D.at<double>(1, 1) = 170.0;
    figure_points_3D.at<double>(1, 2) = -135.0;
    figure_points_3D.at<double>(2, 0) = 0.0;       // Nose tip
    figure_points_3D.at<double>(2, 1) = 0.0;
    figure_points_3D.at<double>(2, 2) = 0.0;
    figure_points_3D.at<double>(3, 0) = 150.0;     // Right mouth corner
    figure_points_3D.at<double>(3, 1) = -150.0;
    figure_points_3D.at<double>(3, 2) = -125.0;
    figure_points_3D.at<double>(4, 0) = -150.0;    // Left mouth corner
    figure_points_3D.at<double>(4, 1) = -150.0;
    figure_points_3D.at<double>(4, 2) = -125.0;
    figure_points_3D.at<double>(5, 0) = 0.0;       // Chin
    figure_points_3D.at<double>(5, 1) = -330.0;
    figure_points_3D.at<double>(5, 2) = -65.0;
    return figure_points_3D;
}

cv::Mat get_camera_matrix()
{
    cv::Mat matrix_camera = cv::Mat::eye(3, 3, CV_64F);
    matrix_camera.at<double>(0, 0) = 1013.80634;
    matrix_camera.at<double>(0, 2) = 632.511658;
    matrix_camera.at<double>(1, 1) = 1020.62616;
    matrix_camera.at<double>(1, 2) = 259.604004;
    return matrix_camera;
}

cv::Mat get_distortion_coeff()
{
    cv::Mat distortion_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    distortion_coeffs.at<double>(0, 0) = 0.05955474;
    distortion_coeffs.at<double>(0, 1) = -0.6827085;
    distortion_coeffs.at<double>(0, 2) = -0.03125953;
    distortion_coeffs.at<double>(0, 3) = -0.00254411;
    distortion_coeffs.at<double>(0, 4) = 1.316122;
    return distortion_coeffs;
}

void estimate_chin(cv::Mat& image_points_2D)
{
    cv::Point eye_midpoint((image_points_2D.at<double>(0, 0) + image_points_2D.at<double>(1, 0)) / 2, (image_points_2D.at<double>(0, 1) + image_points_2D.at<double>(1, 1)) / 2);
    cv::Point mouth_midpoint((image_points_2D.at<double>(3, 0) + image_points_2D.at<double>(4, 0)) / 2, (image_points_2D.at<double>(3, 1) + image_points_2D.at<double>(4, 1)) / 2);

    double slope;
    double intercept;

    double chin_x = 0;
    double chin_y = mouth_midpoint.y - (eye_midpoint.y - mouth_midpoint.y) / 2;

    if ((eye_midpoint.x - mouth_midpoint.x) == 0)
    {
        chin_x = mouth_midpoint.x;

    }
    else
    {
        // 두 중간 점을 지나는 직선 계산
        slope = (eye_midpoint.y - mouth_midpoint.y) / (eye_midpoint.x - mouth_midpoint.x);
        intercept = eye_midpoint.y - slope * eye_midpoint.x;

        if (slope == std::numeric_limits<double>::infinity() || intercept == std::numeric_limits<double>::infinity()) {
            chin_x = mouth_midpoint.x;
        }
        else {
            chin_x = (chin_y - intercept) / slope;
        }
    }
    image_points_2D.at<double>(5, 0) = int(chin_x);
    image_points_2D.at<double>(5, 1) = int(chin_y);
}


bool visualize(cv::Mat& image, const cv::Mat& faces, cv::Mat& image_points_2D, double fps = 0.0)
{
    bool res = false;
    static cv::Scalar box_color{ 0, 255, 0 };
    static cv::Scalar text_color{ 0, 255, 0 };

    std::vector<cv::Scalar> landmark_color = {
        cv::Scalar(255, 0, 0),  // right eye
        cv::Scalar(0, 0, 255),  // left eye
        cv::Scalar(0, 255, 0),  // nose tip
        cv::Scalar(255, 0, 255),// right mouth corner
        cv::Scalar(0, 255, 255) // left mouth corner
    };

    if (fps != 0.0) {
        cv::putText(image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0)) * 2;
        int y1 = static_cast<int>(faces.at<float>(i, 1)) * 2;
        int w = static_cast<int>(faces.at<float>(i, 2)) * 2;
        int h = static_cast<int>(faces.at<float>(i, 3)) * 2;
        cv::rectangle(image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);
        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j)
        {
            res = true;
            int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)) * 2, y = static_cast<int>(faces.at<float>(i, 2 * j + 5)) * 2;
            cv::circle(image, cv::Point(x, y), 2, landmark_color[j], 2);
            //std::cout << x << "," << y << std::endl;
            image_points_2D.at<double>(j, 0) = static_cast<double>(x);
            image_points_2D.at<double>(j, 1) = static_cast<double>(y);
        }
    }
 
    return res;
}

std::vector<cv::Point2d> get_pose_points(cv::Mat& image_points_2D, cv::Mat& vector_rotation, cv::Mat& vector_translation, cv::Mat& camera_matrix, cv::Mat& distortion_coeff)
{
    std::vector<cv::Point2d> pose_points;
    cv::Mat nose_end_point3D = (cv::Mat_<double>(1, 3) << 0, 0, 1000.0);
    cv::Mat nose_end_point2D;
    cv::projectPoints(nose_end_point3D, vector_rotation, vector_translation, camera_matrix, distortion_coeff, nose_end_point2D);

    cv::Point2d point1(image_points_2D.at<double>(2, 0), image_points_2D.at<double>(2, 1));
    cv::Point2d point2(nose_end_point2D.at<double>(0, 0), nose_end_point2D.at<double>(0, 1));

    pose_points.push_back(point1);
    pose_points.push_back(point2);
    return pose_points;
}

double clip_avg(std::vector<int> xdiff_vector)
{
    double xdiff_vector_sum = std::accumulate(xdiff_vector.begin(), xdiff_vector.end(), 0.0);
    double avg = xdiff_vector_sum / xdiff_vector.size();

    if (avg >= 20) {
        avg += -20;
    }
    else if (avg > -20 && avg < 20) {
        avg = 0;
    }
    else {
        avg += 20;
    }
    return avg / 2;
}