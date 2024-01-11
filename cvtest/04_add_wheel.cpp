#include <opencv2/opencv.hpp>
#include <vector>


cv::Mat overlayTransparent(const cv::Mat& background_img, const cv::Mat& img_to_overlay_t, int x, int y)
{
    cv::Mat bg_img = background_img.clone();
    cv::Mat overlay_img = img_to_overlay_t.clone();

    std::vector<cv::Mat> channels;
    cv::split(overlay_img, channels);
    cv::Mat overlay_color;
    cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, overlay_color);

    cv::Mat mask;
    cv::medianBlur(channels[3], mask, 5);

    cv::Rect roi(x, y, overlay_color.cols, overlay_color.rows);
    cv::Mat roi_bg = bg_img(roi).clone();
    cv::Mat bitnot_mask;
    cv::bitwise_not(mask, bitnot_mask);
    cv::bitwise_and(roi_bg, roi_bg, roi_bg, bitnot_mask);
    cv::bitwise_and(overlay_color, overlay_color, overlay_color, mask);
    cv::add(roi_bg, overlay_color, roi_bg);
    roi_bg.copyTo(bg_img(roi));

    return bg_img;
}


std::vector<cv::Rect> getSkinRegions(cv::Mat frame, cv::Rect roi, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Mat kernel)
{
    cv::Mat frameHSV;
    cv::Mat croppedFrame = frame(roi);
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

    return skinRegions;
}


int main() {
    cv::VideoCapture cap(0);  // 웹캠을 열기 위한 VideoCapture 객체 생성
    if (!cap.isOpened()) {
        std::cout << "웹캠을 열 수 없습니다." << std::endl;
        return -1;
    }
    cv::Mat wheelImage = cv::imread("wheel.png", cv::IMREAD_UNCHANGED);
    cv::resize(wheelImage, wheelImage, cv::Size(240, 240));
    cv::Mat frame;
    cv::Rect roi(80, 240, 480, 240);
    cv::Rect wheelRoi(270, 310, 100, 100);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::Scalar lowerBound(0, 48, 80);  // 살색의 하한값
    cv::Scalar upperBound(20, 255, 255);  // 살색의 상한값




    while (cap.read(frame)) {
        cv::flip(frame, frame, 1);
        std::vector<cv::Rect> skinRegions = getSkinRegions(frame, roi, lowerBound, upperBound, kernel);

        frame = overlayTransparent(frame, wheelImage, 200, 240);
        for (const cv::Rect& rect : skinRegions) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);  // 사각형 그리기
        }

        cv::imshow("Skin Detection", frame);
        if (cv::waitKey(10) == 27) {  // ESC 키를 누르면 종료
            break;
        }

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
