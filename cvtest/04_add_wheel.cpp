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

    return skinRegions;
}


int main() {
    cv::VideoCapture cap(0);  // ��ķ�� ���� ���� VideoCapture ��ü ����
    if (!cap.isOpened()) {
        std::cout << "��ķ�� �� �� �����ϴ�." << std::endl;
        return -1;
    }
    cv::Mat wheelImage = cv::imread("wheel.png", cv::IMREAD_UNCHANGED);
    cv::resize(wheelImage, wheelImage, cv::Size(240, 240));
    cv::Mat frame;
    cv::Rect roi(80, 240, 480, 240);
    cv::Rect wheelRoi(270, 310, 100, 100);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::Scalar lowerBound(0, 48, 80);  // ����� ���Ѱ�
    cv::Scalar upperBound(20, 255, 255);  // ����� ���Ѱ�




    while (cap.read(frame)) {
        cv::flip(frame, frame, 1);
        std::vector<cv::Rect> skinRegions = getSkinRegions(frame, roi, lowerBound, upperBound, kernel);

        frame = overlayTransparent(frame, wheelImage, 200, 240);
        for (const cv::Rect& rect : skinRegions) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);  // �簢�� �׸���
        }

        cv::imshow("Skin Detection", frame);
        if (cv::waitKey(10) == 27) {  // ESC Ű�� ������ ����
            break;
        }

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
