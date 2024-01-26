#include <opencv2/opencv.hpp>

int main() {
    // �̹��� ���� �ε�
    cv::Mat image = cv::imread("rotated_img.jpg");
    cv::Mat rot_img, affine_img, inv_affine_img;
    // ȸ�� �߽� ��ǥ ���
    cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
    float theta = 90;
    float scale = 1.3;



    // 90�� �������� ��ȯ ��� ����
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, theta, scale);

    // �̹��� ȸ��
    cv::warpAffine(image, rot_img, rot_mat, image.size());



    /*
    // ��ȯ ���� ��ǥ
    std::vector<cv::Point2f> src(4);
    src[0] = cv::Point2f(0, 0);
    src[1] = cv::Point2f(0, 1);
    src[2] = cv::Point2f(1, 0);
    src[3] = cv::Point2f(1, 1);

    // ��ȯ ���� ��ǥ
    std::vector<cv::Point2f> dst(4);
    dst[0] = cv::Point2f(0, 0);
    dst[1] = cv::Point2f(0, 0.5);
    dst[2] = cv::Point2f(0.5, 0);
    dst[3] = cv::Point2f(1, 1);
    */
    cv::Point2f srcTri[3];
    srcTri[0] = cv::Point2f(0.f, 0.f);
    srcTri[1] = cv::Point2f(image.cols - 1.f, 0.f);
    srcTri[2] = cv::Point2f(0.f, image.rows - 1.f);

    cv::Point2f dstTri[3];
    dstTri[0] = cv::Point2f(0.f, image.rows * 0.33f);
    dstTri[1] = cv::Point2f(image.cols * 0.85f, image.rows * 0.25f);
    dstTri[2] = cv::Point2f(image.cols * 0.15f, image.rows * 0.7f);

    // ������ ��ȯ ��� ���
    cv::Mat affineTransform = cv::getAffineTransform(srcTri, dstTri);
    cv::Mat inv_affine_tranform;
    cv::invertAffineTransform(affineTransform, inv_affine_tranform);


    cv::warpAffine(image, affine_img, affineTransform, image.size());
    cv::warpAffine(image, inv_affine_img, inv_affine_tranform, image.size());







    //cv::Mat rot_img, affine_img, inv_affine_img;



    // �̹��� ���
    cv::imshow("image", image);
    cv::imshow("rot_img", rot_img);

    cv::imshow("affine_img", affine_img);
    cv::imshow("inv_affine_img", inv_affine_img);

    cv::waitKey(0);






    return 0;
}
