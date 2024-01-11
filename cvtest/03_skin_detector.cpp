#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::VideoCapture cap(0);  // ��ķ�� ���� ���� VideoCapture ��ü ����
    if (!cap.isOpened()) {
        std::cout << "��ķ�� �� �� �����ϴ�." << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Rect roi(80, 240, 480, 240);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::Scalar lowerBound(0, 48, 80);  // ����� ���Ѱ�
    cv::Scalar upperBound(20, 255, 255);  // ����� ���Ѱ�

    while (cap.read(frame)) {
        cv::Mat frameHSV;
        cv::Mat croppedFrame = frame(roi).clone();
        cv::cvtColor(croppedFrame, frameHSV, cv::COLOR_BGR2HSV);  // BGR�� HSV�� ��ȯ

        cv::Mat skinMask;
        cv::inRange(frameHSV, lowerBound, upperBound, skinMask);  // ��� ������ ���ϴ� �ȼ��� ����ũ�� ����

        cv::Mat skin;
        cv::bitwise_and(croppedFrame, croppedFrame, skin, skinMask);  // ���� �̹����� ����ũ�� �̿��Ͽ� ��� ���� ����

        cv::Mat skinGray;
        cv::cvtColor(skin, skinGray, cv::COLOR_BGR2GRAY);  // ������ ��� ������ �׷��̽����Ϸ� ��ȯ

        cv::Mat closing;
        cv::morphologyEx(skinGray, closing, cv::MORPH_CLOSE, kernel);


        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(closing, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);  // ������ ����

        std::vector<cv::Rect> skinRegions;
        for (const std::vector<cv::Point>& contour : contours) {
            cv::Rect rect = cv::boundingRect(contour);  // �������� ���δ� �簢�� ����
            auto area = rect.width * rect.height;
            if ((area >= 40 * 40) && (area <= 300 * 200))
            {
                rect.x += roi.x;
                rect.y += roi.y;
                skinRegions.push_back(rect);  // �簢���� ���Ϳ� �߰�
            }
        }

        for (const cv::Rect& rect : skinRegions) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);  // �簢�� �׸���
        }
        frame(roi) = croppedFrame;

        cv::imshow("skin", skin);
        cv::imshow("skinGray", skinGray);
        cv::imshow("closing", closing);
        cv::imshow("croppedFrame", croppedFrame);
        cv::imshow("Skin Detection", frame);
        if (cv::waitKey(10) == 27) {  // ESC Ű�� ������ ����
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
