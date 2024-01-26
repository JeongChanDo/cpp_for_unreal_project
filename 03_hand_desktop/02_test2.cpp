#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("testimg.jpg");


    float theta = -30;
    cv::Rect box(260, 240, 120, 120);

    float scale = 2.6;

    // 회전 중심점 계산
    cv::Point2f center(box.x + box.width / 2, box.y + box.height / 2);

    // 회전 변환 행렬 계산
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, theta, 1.0);


    // 원래 detection 꼭지점
    cv::Point2f originaDetPoints[4] = {
    cv::Point2f(box.x, box.y),
    cv::Point2f(box.x + box.width, box.y),
    cv::Point2f(box.x + box.width, box.y + box.height),
    cv::Point2f(box.x, box.y + box.height)
    };

    // 회전된 상자의 꼭지점 계산
    cv::Point2f startRotatedPoints[4];

    // 목표 꼭지점(블라즈 핸드는 256이므로
    cv::Point2f targetRotatedPoint[4] = {
        cv::Point2f(0, 0),
        cv::Point2f(256, 0),
        cv::Point2f(256, 256),
        cv::Point2f(0, 256)

    };

    for (int i = 0; i < 4; i++)
    {
        cv::Point2f startRotatedPoint = originaDetPoints[i] - center;
        float x = startRotatedPoint.x * std::cos(theta * CV_PI / 180) - startRotatedPoint.y * std::sin(theta * CV_PI / 180);
        float y = startRotatedPoint.x * std::sin(theta * CV_PI / 180) + startRotatedPoint.y * std::cos(theta * CV_PI / 180);
        x = x * scale;
        y = y * scale;
        startRotatedPoints[i] = cv::Point2f(x, y) + center;
    }

    // 회전된 상자 그리기
    for (int i = 0; i < 4; i++) {
        cv::line(img, startRotatedPoints[i], startRotatedPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }


    // 어파인 변환 행렬 계산
    cv::Mat affineTransform_Mat = cv::getAffineTransform(startRotatedPoints, targetRotatedPoint);
    cv::Mat inv_affine_tranform_Mat;
    cv::invertAffineTransform(affineTransform_Mat, inv_affine_tranform_Mat);
    
    // 역/정 어파인변환 행렬로 이미지 변환
    cv::Mat affine_img;
    cv::Mat inv_affine_img;
    cv::warpAffine(img, affine_img, affineTransform_Mat, cv::Size(256, 256));
    cv::warpAffine(affine_img, inv_affine_img, inv_affine_tranform_Mat, img.size());


    // 결과 이미지 출력
    cv::imshow("img", img);
    cv::imshow("img2affine_img", affine_img);
    cv::imshow("affine_img2inv_affine_img", inv_affine_img);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
