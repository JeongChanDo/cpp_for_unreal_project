#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::VideoCapture cap(0);  // 웹캠을 열기 위한 VideoCapture 객체 생성
    if (!cap.isOpened()) {
        std::cout << "웹캠을 열 수 없습니다." << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Rect roi(80, 240, 480, 240);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::Scalar lowerBound(0, 48, 80);  // 살색의 하한값
    cv::Scalar upperBound(20, 255, 255);  // 살색의 상한값

    while (cap.read(frame)) {
        cv::Mat frameHSV;
        cv::Mat croppedFrame = frame(roi).clone();
        cv::cvtColor(croppedFrame, frameHSV, cv::COLOR_BGR2HSV);  // BGR을 HSV로 변환

        cv::Mat skinMask;
        cv::inRange(frameHSV, lowerBound, upperBound, skinMask);  // 살색 범위에 속하는 픽셀을 마스크로 생성

        cv::Mat skin;
        cv::bitwise_and(croppedFrame, croppedFrame, skin, skinMask);  // 원본 이미지와 마스크를 이용하여 살색 영역 추출

        cv::Mat skinGray;
        cv::cvtColor(skin, skinGray, cv::COLOR_BGR2GRAY);  // 추출한 살색 영역을 그레이스케일로 변환

        cv::Mat closing;
        cv::morphologyEx(skinGray, closing, cv::MORPH_CLOSE, kernel);


        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(closing, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);  // 윤곽선 검출

        std::vector<cv::Rect> skinRegions;
        for (const std::vector<cv::Point>& contour : contours) {
            cv::Rect rect = cv::boundingRect(contour);  // 윤곽선을 감싸는 사각형 생성
            auto area = rect.width * rect.height;
            if ((area >= 40 * 40) && (area <= 300 * 200))
            {
                rect.x += roi.x;
                rect.y += roi.y;
                skinRegions.push_back(rect);  // 사각형을 벡터에 추가
            }
        }

        for (const cv::Rect& rect : skinRegions) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);  // 사각형 그리기
        }
        frame(roi) = croppedFrame;

        cv::imshow("skin", skin);
        cv::imshow("skinGray", skinGray);
        cv::imshow("closing", closing);
        cv::imshow("croppedFrame", croppedFrame);
        cv::imshow("Skin Detection", frame);
        if (cv::waitKey(10) == 27) {  // ESC 키를 누르면 종료
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
