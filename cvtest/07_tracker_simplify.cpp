#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

struct TrackedRect {
    cv::Rect rect;
    int lifespan;
};

float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2);
std::map<int, TrackedRect> getTrackedRects(std::map<int, TrackedRect> trackedRects, std::vector<cv::Rect> skinRegions, uint32_t& nextId);
std::map<int, TrackedRect> getResRects(std::map<int, TrackedRect> trackedRects);
void drawTrackedRect(cv::Mat frame, std::pair<int, TrackedRect> trackedRect);
cv::Mat overlayTransparent(const cv::Mat& background_img, const cv::Mat& img_to_overlay_t, int x, int y);
std::vector<cv::Rect> getSkinRegions(cv::Mat frame, cv::Rect roi, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Mat kernel);


int main()
{
    cv::VideoCapture cap(0);  // ��ķ�� ���� ���� VideoCapture ��ü ����
    if (!cap.isOpened()) {
        std::cout << "��ķ�� �� �� �����ϴ�." << std::endl;
        return -1;
    }

    uint32_t nextId = 0; //tracking id
    std::map<int, TrackedRect> trackedRects; // map to store tracked rectangles

    cv::Mat wheelImage = cv::imread("wheel.png", cv::IMREAD_UNCHANGED);
    cv::resize(wheelImage, wheelImage, cv::Size(240, 240));

    cv::Mat frame;
    cv::Rect handRoi(120, 240, 520, 240);
    cv::Rect objDetectRoi(120, 40, 520, 440);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::Scalar lowerBound(0, 48, 80);  // ����� ���Ѱ�
    cv::Scalar upperBound(20, 255, 255);  // ����� ���Ѱ�



    while (cap.read(frame)) {

        cv::flip(frame, frame, 1);
        std::vector<cv::Rect> skinRegions = getSkinRegions(frame, handRoi, lowerBound, upperBound, kernel);

        frame = overlayTransparent(frame, wheelImage, 200, 240);

        //skinRegions�� trackedRects ����(���, �����ֱ� Ȯ��, ����)
        trackedRects = getTrackedRects(trackedRects, skinRegions, nextId);


        // trackedRect�� 2�� �̻��, ���� ū trackedRect 2�� ��� �ð�ȭ
        std::map<int, TrackedRect> resRects;
        if (trackedRects.size() >= 2)
        {
            //���� ū �ڽ� 2�� ��ȯ
            resRects = getResRects(trackedRects);
        }
        else
        {
            resRects = trackedRects;
        }

        // trackedRects �ð�ȭ
        for (auto& trackedRect : resRects)
        {
            drawTrackedRect(frame, trackedRect);
        }

        cv::rectangle(frame, handRoi, cv::Scalar(255, 0, 0), 1);  // �簢�� �׸���
        cv::rectangle(frame, objDetectRoi, cv::Scalar(0, 255, 0), 1);  // �簢�� �׸���

        cv::imshow("Skin Detection", frame);
        if (cv::waitKey(10) == 27) {  // ESC Ű�� ������ ����
            break;
        }

    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
}



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



std::map<int, TrackedRect> getTrackedRects(std::map<int, TrackedRect> trackedRects, std::vector<cv::Rect> skinRegions, uint32_t& nextId)
{
    std::vector<int> foundIndex;

    for (const cv::Rect& skinRegion : skinRegions)
    {
        bool foundMatch = false;
        for (auto& trackedRect : trackedRects)
        {
            // Calculate IOU between detected rectangle and tracked rectangle
            float iou = calculateIOU(skinRegion, trackedRect.second.rect);

            // If IOU is above a threshold, update the tracked rectangle
            if (iou > 0.4) {
                trackedRect.second.rect = skinRegion;
                foundMatch = true;
                foundIndex.push_back(trackedRect.first);
                break;
            }
        }

        // If no match found, add new tracked rectangle
        if (!foundMatch) {
            trackedRects[nextId].rect = skinRegion;
            trackedRects[nextId].lifespan = 5;
            nextId++;
        }
    }

    //if trackedRect not found, lifespan -1
    for (auto& trackedRect : trackedRects)
    {
        bool foundMatch = false;
        for (auto& id : foundIndex)
        {
            if (id == trackedRect.first)
            {
                foundMatch = true;
                trackedRect.second.lifespan = 5; //ã���� 5�� �ٽ� ����
                break;
            }
        }
        if (foundMatch == false)
        {
            trackedRect.second.lifespan -= 1;
        }
    }

    // lifespan == 0�� trackedRect ����
    for (auto& trackedRect : trackedRects)
    {
        if (trackedRect.second.lifespan == 0)
        {
            trackedRects.erase(trackedRect.first);
        }
    }
    return trackedRects;
}


std::map<int, TrackedRect> getResRects(std::map<int, TrackedRect> trackedRects)
{
    std::map<int, TrackedRect> resRects;

    std::vector<std::pair<int, TrackedRect>> sortedTracedRects(trackedRects.begin(), trackedRects.end());
    std::sort(
        sortedTracedRects.begin(), sortedTracedRects.end(), [](const std::pair<int, TrackedRect>& a, const std::pair<int, TrackedRect>& b) {
            return a.second.rect.width * a.second.rect.height > b.second.rect.width * b.second.rect.height;
        });

    int key1 = sortedTracedRects[0].first;
    int key2 = sortedTracedRects[1].first;
    resRects[key1] = trackedRects[key1];
    resRects[key2] = trackedRects[key2];

    return resRects;
}


void drawTrackedRect(cv::Mat frame, std::pair<int, TrackedRect> trackedRect)
{
    cv::putText(
        frame, cv::String(std::to_string(trackedRect.first)),
        cv::Point(int(trackedRect.second.rect.x), int(trackedRect.second.rect.y - 20)),
        cv::FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0)
    );

    cv::putText(
        frame, cv::String(std::to_string(trackedRect.second.lifespan)),
        cv::Point(int(trackedRect.second.rect.x + 100), int(trackedRect.second.rect.y - 20)),
        cv::FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255)
    );
    cv::rectangle(frame, trackedRect.second.rect, cv::Scalar(0, 255, 0), 2);  // �簢�� �׸���

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
        if ((area >= 70 * 70) && (area <= 300 * 200))
        {
            rect.x += roi.x;
            rect.y += roi.y;
            skinRegions.push_back(rect);  // �簢���� ���Ϳ� �߰�
        }
    }

    return skinRegions;
}