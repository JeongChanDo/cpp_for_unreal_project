#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("testimg.jpg");


    float theta = -30;
    cv::Rect box(260, 240, 120, 120);

    float scale = 2.6;

    // ȸ�� �߽��� ���
    cv::Point2f center(box.x + box.width / 2, box.y + box.height / 2);

    // ȸ�� ��ȯ ��� ���
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, theta, 1.0);


    // ���� detection ������
    cv::Point2f originaDetPoints[4] = {
    cv::Point2f(box.x, box.y),
    cv::Point2f(box.x + box.width, box.y),
    cv::Point2f(box.x + box.width, box.y + box.height),
    cv::Point2f(box.x, box.y + box.height)
    };

    // ȸ���� ������ ������ ���
    cv::Point2f startRotatedPoints[4];

    // ��ǥ ������(����� �ڵ�� 256�̹Ƿ�
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

    // ȸ���� ���� �׸���
    for (int i = 0; i < 4; i++) {
        cv::line(img, startRotatedPoints[i], startRotatedPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }


    // ������ ��ȯ ��� ���
    cv::Mat affineTransform_Mat = cv::getAffineTransform(startRotatedPoints, targetRotatedPoint);
    cv::Mat inv_affine_tranform_Mat;
    cv::invertAffineTransform(affineTransform_Mat, inv_affine_tranform_Mat);
    
    // ��/�� �����κ�ȯ ��ķ� �̹��� ��ȯ
    cv::Mat affine_img;
    cv::Mat inv_affine_img;
    cv::warpAffine(img, affine_img, affineTransform_Mat, cv::Size(256, 256));
    cv::warpAffine(affine_img, inv_affine_img, inv_affine_tranform_Mat, img.size());


    // ��� �̹��� ���
    cv::imshow("img", img);
    cv::imshow("img2affine_img", affine_img);
    cv::imshow("affine_img2inv_affine_img", inv_affine_img);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
