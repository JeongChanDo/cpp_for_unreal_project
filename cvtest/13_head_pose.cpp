#include "head_pose.h"
#include <vector>



int main()
{

    cv::VideoCapture cap(0);  // 웹캠을 열기 위한 VideoCapture 객체 생성
    if (!cap.isOpened()) {
        std::cout << "웹캠을 열 수 없습니다." << std::endl;
        return -1;
    }

    cv::Mat frame;

    cv::Rect head_roi(120, 0, 400, 400);
    cv::Size head_image_size(int(head_roi.width / 2), int(head_roi.height / 2));
    std::vector<int> xdiff_vector;
    const int backend_id = str2backend.at("opencv");
    const int target_id = str2target.at("cpu");

    cv::Mat figure_points_3D = get_figure_points_3D();
    cv::Mat camera_matrix = get_camera_matrix();
    cv::Mat distortion_coeff = get_distortion_coeff();
    cv::Mat vector_rotation = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat vector_translation = (cv::Mat_<double>(3, 1) << 0, 0, 0);

    YuNet model(
        "yunet.onnx",
        head_image_size,
        0.6,
        0.3,
        3000,
        backend_id,
        target_id
    );

    while (cap.read(frame))
    {
        cv::Mat head_area = frame(head_roi);
        model.setInputSize(head_image_size);

        auto tick_meter = cv::TickMeter();
        tick_meter.start();
        cv::Mat resized_head;
        cv::resize(head_area, resized_head, head_image_size);
        cv::Mat faces = model.infer(resized_head);
        tick_meter.stop();

        cv::Mat image_points_2D = get_image_points_2D();
        if (visualize(head_area, faces, image_points_2D, (float)tick_meter.getFPS()))
        {
            estimate_chin(image_points_2D);
            cv::circle(head_area, cv::Point(int(image_points_2D.at<double>(5, 0)), int(image_points_2D.at<double>(5, 1))), 2, cv::Scalar(255, 255, 255), 2);

        }

        if (cv::solvePnPRansac(
                figure_points_3D,
                image_points_2D,
                camera_matrix,
                distortion_coeff,
                vector_rotation,
                vector_translation
        ))
        {
            std::vector<cv::Point2d> pose_points = get_pose_points(image_points_2D, vector_rotation, vector_translation, camera_matrix, distortion_coeff);
            cv::line(head_area, pose_points[0], pose_points[1], cv::Scalar(255, 255, 255), 2);
            int x_diff = pose_points[0].x - pose_points[1].x;
            xdiff_vector.push_back(x_diff);
            cv::putText(head_area, cv::format("%d", x_diff), cv::Point(pose_points[0].x - 10, pose_points[0].y - 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0));
            if (xdiff_vector.size() == 41)
            {
                xdiff_vector.erase(xdiff_vector.begin());
                double cliped_xdiff = clip_avg(xdiff_vector);
                cv::putText(head_area, cv::format("%.1f", cliped_xdiff), cv::Point(pose_points[0].x + 40, pose_points[0].y - 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 2);
            }
        }

        // Visualize in a new window
        cv::imshow("YuNet Demo", head_area);
        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27) {  // ESC 키를 누르면 종료
            break;
        }

    }
    cap.release();
    cv::destroyAllWindows();

}