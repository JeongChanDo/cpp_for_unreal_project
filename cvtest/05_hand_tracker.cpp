#include <opencv2/opencv.hpp>
#include <vector>
#include <map>


// Function to calculate Intersection over Union (IOU) between two rectangles
float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = rect1.width * rect1.height + rect2.width * rect2.height - intersectionArea;

    return static_cast<float>(intersectionArea) / unionArea;
}


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


int main() 
{
    cv::VideoCapture cap(0);  // ��ķ�� ���� ���� VideoCapture ��ü ����
    if (!cap.isOpened()) {
        std::cout << "��ķ�� �� �� �����ϴ�." << std::endl;
        return -1;
    }

    uint32_t nextId = 0; //tracking id
    std::map<int, cv::Rect> trackedRects; // map to store tracked rectangles

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
        for (const cv::Rect& skinRegion : skinRegions)
        {
            bool foundMatch = false;

            for (auto& trackedRect : trackedRects)
            {
                // Calculate IOU between detected rectangle and tracked rectangle
                float iou = calculateIOU(skinRegion, trackedRect.second);

                // If IOU is above a threshold, update the tracked rectangle
                if (iou > 0.4) {
                    trackedRect.second = skinRegion;
                    foundMatch = true;
                    break;
                }

            }

            // If no match found, add new tracked rectangle
            if (!foundMatch) {
                trackedRects[nextId] = skinRegion;
                nextId++;
            }

        }

          
        for (auto& trackedRect : trackedRects)
        {
            cv::putText(
                frame, cv::String(std::to_string(trackedRect.first)), 
                cv::Point(int(trackedRect.second.x), int(trackedRect.second.y - 20)),
                cv::FONT_HERSHEY_SIMPLEX, 2, (125, 125, 125)
            );
            cv::rectangle(frame, trackedRect.second, cv::Scalar(0, 255, 0), 2);  // �簢�� �׸���
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
