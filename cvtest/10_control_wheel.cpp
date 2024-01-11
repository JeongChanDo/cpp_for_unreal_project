#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

struct WheelInfo {
    float angle;
    cv::Point2d leftPoint;
    cv::Point2d rightPoint;
    float radius;
};

struct DetectResult {
    float score;
    int x;
    int y;
    int w;
    int h;
};

struct TrackedRect {
    cv::Rect rect;
    int lifespan;
    bool isHandDetected;
};

// color tracking
float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2);
std::map<int, TrackedRect> getTrackedRects(std::map<int, TrackedRect> trackedRects, std::vector<cv::Rect> skinRegions, uint32_t& nextId);
std::map<int, TrackedRect> getResRects(std::map<int, TrackedRect> trackedRects);
void drawTrackedRect(cv::Mat frame, std::pair<int, TrackedRect> trackedRect);
cv::Mat overlayTransparent(const cv::Mat& background_img, const cv::Mat& img_to_overlay_t, int x, int y);
std::vector<cv::Rect> getSkinRegions(cv::Mat frame, cv::Rect roi, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Mat kernel);

// hand detect
float sigmoid(float x);
DetectResult getDetectResult(cv::Mat& frame, cv::Mat regressor, cv::Mat classificator, int stride, int anchor_count, int column, int row, int anchor, int offset, cv::Size detectInputSize, cv::Size detectOutputSize);
std::vector<DetectResult> getDetectResults(cv::Mat frame, cv::dnn::Net net, cv::Size detectInputSize, cv::Size detectOutputSize);
void drawDetectResult(cv::Mat frame, DetectResult res, cv::Rect objDetectRoi);
void checkIsHand(std::map<int, TrackedRect>& trackedRects, std::vector<DetectResult>& detectResults, cv::Rect objDetectRoi);

// control
cv::Mat drawAndGetRotatedPoints(cv::Mat frame, cv::Point2f wheelStartPoint,  cv::Mat Points, cv::Mat rotationMat);
void wheelCheck(cv::Mat frame, bool& wheelLeftChecked, bool& wheelRightChecked, cv::Point2f wheelStartPoint, cv::Mat rotatedPoints, std::map<int, TrackedRect> resRects);
float getWheelAngle(std::map<int, TrackedRect> resRects);

int main()
{
    cv::VideoCapture cap(0);  // 웹캠을 열기 위한 VideoCapture 객체 생성
    if (!cap.isOpened()) {
        std::cout << "웹캠을 열 수 없습니다." << std::endl;
        return -1;
    }
    cv::dnn::Net net = cv::dnn::readNetFromONNX("c:/palm_detection.onnx");

    uint32_t nextId = 0; //tracking id
    std::map<int, TrackedRect> trackedRects; // map to store tracked rectangles

    cv::Mat wheelImage = cv::imread("wheel.png", cv::IMREAD_UNCHANGED);
    cv::resize(wheelImage, wheelImage, cv::Size(240, 240));

    cv::Mat frame;
    cv::Rect handRoi(120, 240, 520, 240);
    cv::Rect objDetectRoi(120, 40, 520, 440);
    cv::Size detectInputSize(128, 128);
    cv::Size detectOutputSize(objDetectRoi.width, objDetectRoi.height);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::Scalar lowerBound(0, 48, 80);  // 살색의 하한값
    cv::Scalar upperBound(20, 255, 255);  // 살색의 상한값



    //휠제어 변수들
    float angle = 0;
    cv::Point2f center(120, 120);
    cv::Point2f leftPoint(0, 120);
    cv::Point2f rightPoint(240, 120);
    // 좌표를 2차원 행렬로 변환
    cv::Mat points(3, 1, CV_64FC2);
    points.at<cv::Vec2d>(0, 0) = cv::Vec2d(leftPoint.x, leftPoint.y);
    points.at<cv::Vec2d>(1, 0) = cv::Vec2d(rightPoint.x, rightPoint.y);
    points.at<cv::Vec2d>(2, 0) = cv::Vec2d(center.x, center.y);
    cv::Point2f wheelStartPoint(200, 240);


    while (cap.read(frame)) {

        cv::flip(frame, frame, 1);
        std::vector<cv::Rect> skinRegions = getSkinRegions(frame, handRoi, lowerBound, upperBound, kernel);


        //skinRegions로 trackedRects 구함(등록, 생명주기 확인, 제거)
        trackedRects = getTrackedRects(trackedRects, skinRegions, nextId);

        // 객체 탐지 및 결과 반환
        cv::Mat detectFrame = frame(objDetectRoi).clone();
        std::vector<DetectResult> detectResults = getDetectResults(detectFrame, net, detectInputSize, detectOutputSize);

        // detectResults 시각화
        for (auto& detectResult : detectResults)
        {
            drawDetectResult(frame, detectResult, objDetectRoi);
        }


        // trackedRects에 있는 것들이 detectResults로 손인지 확인
        checkIsHand(trackedRects, detectResults, objDetectRoi);

        // trackedRect가 2개 이상시, 가장 큰 trackedRect 2개 골라서 시각화
        std::map<int, TrackedRect> resRects;
        if (trackedRects.size() >= 2)
        {
            //가장 큰 박스 2개 반환
            resRects = getResRects(trackedRects);
        }
        else
        {
            resRects = trackedRects;
        }

        // trackedRects 시각화
        for (auto& trackedRect : resRects)
        {
            drawTrackedRect(frame, trackedRect);
        }

        cv::rectangle(frame, handRoi, cv::Scalar(255, 0, 0), 1);  // 사각형 그리기
        cv::rectangle(frame, objDetectRoi, cv::Scalar(0, 255, 0), 1);  // 사각형 그리기


        cv::Mat rotatedImage = wheelImage.clone();
        cv::Mat rotationMat = cv::getRotationMatrix2D(center, angle, 1.0);

        cv::line(rotatedImage, leftPoint, rightPoint, cv::Scalar(0, 0, 255), 5);
        cv::warpAffine(rotatedImage, rotatedImage, rotationMat, wheelImage.size());


        // 좌표 회전 적용
        cv::Mat rotatedPoints = drawAndGetRotatedPoints(frame, wheelStartPoint, points, rotationMat);

        // 손 2개 있으면 손잡이 확인하고, 회전반영하기
        if (resRects.size() == 2)
        {
            bool wheelLeftChecked = false;
            bool wheelRightChecked = false;

            wheelCheck(frame, wheelLeftChecked, wheelRightChecked, wheelStartPoint, rotatedPoints, resRects);
            if (wheelLeftChecked == true && wheelRightChecked == true)
            {
                angle = getWheelAngle(resRects);
            }
        }


        frame = overlayTransparent(frame, rotatedImage, wheelStartPoint.x, wheelStartPoint.y);
        cv::imshow("Skin Detection", frame);
        cv::imshow("wheelImage", wheelImage);
        cv::imshow("rot img", rotatedImage);
        if (cv::waitKey(10) == 27) {  // ESC 키를 누르면 종료
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
            trackedRects[nextId].isHandDetected = false;
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
                trackedRect.second.lifespan = 5; //찾으면 5로 다시 설정
                break;
            }
        }
        if (foundMatch == false)
        {
            trackedRect.second.lifespan -= 1;
        }
    }

    // lifespan == 0인 trackedRect 삭제
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

    if (trackedRect.second.isHandDetected == true)
    {
        cv::rectangle(frame, trackedRect.second.rect, cv::Scalar(255, 0, 0), 6);  // 사각형 그리기
    }
    cv::rectangle(frame, trackedRect.second.rect, cv::Scalar(0, 255, 0), 2);  // 사각형 그리기
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
        if ((area >= 70 * 70) && (area <= 300 * 200))
        {
            rect.x += roi.x;
            rect.y += roi.y;
            skinRegions.push_back(rect);  // 사각형을 벡터에 추가
        }
    }

    return skinRegions;
}


float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}



DetectResult getDetectResult(cv::Mat& frame, cv::Mat regressor, cv::Mat classificator,
    int stride, int anchor_count, int column, int row, int anchor, int offset,
    cv::Size detectInputSize, cv::Size detectOutputSize) {

    DetectResult res{ 0.0f, 0, 0, 0, 0 };

    int index = (int(row * 128 / stride) + column) * anchor_count + anchor + offset;
    float origin_score = regressor.at<float>(0, index, 0);
    float score = sigmoid(origin_score);
    if (score < 0.5) return res;

    float x = classificator.at<float>(0, index, 0);
    float y = classificator.at<float>(0, index, 1);
    float w = classificator.at<float>(0, index, 2);
    float h = classificator.at<float>(0, index, 3);


    x += (column + 0.5) * stride - w / 2;
    y += (row + 0.5) * stride - h / 2;

    float width_ratio = static_cast<float>(detectOutputSize.width) / static_cast<float>(detectInputSize.width);
    float height_radio = static_cast<float>(detectOutputSize.height) / static_cast<float>(detectInputSize.height);
    res.score = score;
    res.x = int(x * width_ratio);
    res.y = int(y * height_radio);
    res.w = int(w * width_ratio);
    res.h = int(h * height_radio);
    return res;
}


std::vector<DetectResult> getDetectResults(cv::Mat frame, cv::dnn::Net net, cv::Size detectInputSize, cv::Size detectOutputSize)
{
    std::vector<DetectResult> beforeNMSResults;
    std::vector<DetectResult> afterNMSResults;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<cv::Rect> boundingBoxes;

    cv::Mat inputImg;
    cv::resize(frame, inputImg, detectInputSize);
    cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2RGB);

    cv::Mat tensor;
    inputImg.convertTo(tensor, CV_32F, 1 / 127.5, -1.0);
    cv::Mat blob = cv::dnn::blobFromImage(tensor, 1.0, tensor.size(), 0, false, false, CV_32F);
    std::vector<cv::String> outNames(2);
    outNames[0] = "regressors";
    outNames[1] = "classificators";

    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, outNames);

    cv::Mat classificator = outputs[0];
    cv::Mat regressor = outputs[1];


    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            for (int a = 0; a < 2; ++a) {
                DetectResult res = getDetectResult(frame, regressor, classificator, 8, 2, x, y, a, 0, detectInputSize, detectOutputSize);
                if (res.score != 0)
                {
                    beforeNMSResults.push_back(res);
                    boundingBoxes.push_back(cv::Rect(res.x, res.y, res.w, res.h));
                    scores.push_back(res.score);
                }
            }
        }
    }

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            for (int a = 0; a < 6; ++a) {
                DetectResult res = getDetectResult(frame, regressor, classificator, 16, 6, x, y, a, 512, detectInputSize, detectOutputSize);
                if (res.score != 0)
                {
                    beforeNMSResults.push_back(res);
                    boundingBoxes.push_back(cv::Rect(res.x, res.y, res.w, res.h));
                    scores.push_back(res.score);
                }
            }
        }
    }

    cv::dnn::NMSBoxes(boundingBoxes, scores, 0.5, 0.2, indices);

    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        afterNMSResults.push_back(beforeNMSResults[idx]);
    }

    return afterNMSResults;
}

void drawDetectResult(cv::Mat frame, DetectResult res, cv::Rect objDetectRoi)
{
    cv::putText(
        frame, cv::String(std::to_string(res.score)),
        cv::Point(int(res.x + objDetectRoi.x), int(res.y + objDetectRoi.y - 20)),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255)
    );
    cv::rectangle(frame, cv::Rect(res.x + objDetectRoi.x, res.y + objDetectRoi.y, res.w, res.h), cv::Scalar(0, 0, 255), 1);  // 사각형 그리기
}


// 가장 큰 trackedRects 2개(resRects)와 detectResults로 손인지 확인
void checkIsHand(std::map<int, TrackedRect>& trackedRects, std::vector<DetectResult>& detectResults, cv::Rect objDetectRoi)
{
    for (auto& trackedRect : trackedRects)
    {
        for (auto& detectResult : detectResults)
        {
            cv::Rect detectResultRect{ detectResult.x + objDetectRoi.x, detectResult.y + objDetectRoi.y, detectResult.w, detectResult.h };
            float iou = calculateIOU(detectResultRect, trackedRect.second.rect);

            if (iou > 0.4) {
                trackedRect.second.isHandDetected = true;
            }
        }
    }
}

float getWheelAngle(std::map<int, TrackedRect> resRects)
{
    TrackedRect leftRect = resRects.begin()->second;
    TrackedRect rightRect = resRects.begin()->second;

    for (auto& trackedRect : resRects)
    {
        if (trackedRect.second.rect.x < leftRect.rect.x)
            leftRect = trackedRect.second;

        if (trackedRect.second.rect.x > rightRect.rect.x)
            rightRect = trackedRect.second;
    }

    // 두 개의 직사각형 중심점 계산
    cv::Point2f center1(leftRect.rect.x + leftRect.rect.width / 2.0f, leftRect.rect.y + leftRect.rect.height / 2.0f);
    cv::Point2f center2(rightRect.rect.x + rightRect.rect.width / 2.0f, rightRect.rect.y + rightRect.rect.height / 2.0f);

    // 두 중심점을 연결하는 선의 기울기 계산
    float dx = center2.x - center1.x;
    float dy = center2.y - center1.y;
    float angle = (-1) * atan2(dy, dx) * 180.0f / CV_PI;
    return angle;
}


void wheelCheck(cv::Mat frame, bool& wheelLeftChecked, bool& wheelRightChecked, cv::Point2f wheelStartPoint, cv::Mat rotatedPoints, std::map<int, TrackedRect> resRects)
{
        // 회전된 좌표 추출
    cv::Point2f rotatedLeftPoint(rotatedPoints.at<cv::Vec2d>(0, 0)[0], rotatedPoints.at<cv::Vec2d>(0, 0)[1]);
    cv::Point2f rotatedRightPoint(rotatedPoints.at<cv::Vec2d>(1, 0)[0], rotatedPoints.at<cv::Vec2d>(1, 0)[1]);
    cv::Point2f rotatedCenter(rotatedPoints.at<cv::Vec2d>(2, 0)[0], rotatedPoints.at<cv::Vec2d>(2, 0)[1]);

    for (auto& trackedRect : resRects)
    {
        if (trackedRect.second.isHandDetected == false)
            return;

        float circleRadius = 30;
        cv::Mat leftCircleImage = cv::Mat::zeros(640, 480, CV_8UC1);
        cv::circle(leftCircleImage, wheelStartPoint + rotatedLeftPoint, circleRadius, 255, -1);
        cv::Mat leftBoxImage = cv::Mat::zeros(640, 480, CV_8UC1);
        cv::rectangle(leftBoxImage, trackedRect.second.rect, 255, -1);

        // 원과 박스의 교차 영역 계산
        cv::Mat intersectionImageLeft = leftCircleImage & leftBoxImage;

        // 교차 영역의 픽셀 개수 계산
        int intersectionPixelsLeft = cv::countNonZero(intersectionImageLeft);


        float leftOverlapRatio = static_cast<float>(intersectionPixelsLeft) / (circleRadius * circleRadius * CV_PI);

        if (leftOverlapRatio > 0.8)
        {
            cv::circle(frame, wheelStartPoint + rotatedLeftPoint, 30, cv::Scalar(255, 0, 0), -1);
            wheelLeftChecked = true;
        }

        cv::Mat rightCircleImage = cv::Mat::zeros(640, 480, CV_8UC1);
        cv::circle(rightCircleImage, wheelStartPoint + rotatedRightPoint, circleRadius, 255, -1);
        cv::Mat rightBoxImage = cv::Mat::zeros(640, 480, CV_8UC1);
        cv::rectangle(rightBoxImage, trackedRect.second.rect, 255, -1);

        cv::Mat intersectionImageRight = rightCircleImage & rightBoxImage;
        int intersectionPixelsRight = cv::countNonZero(intersectionImageRight);
        float rightOverlapRatio = static_cast<float>(intersectionPixelsRight) / (circleRadius * circleRadius * CV_PI);

        if (rightOverlapRatio > 0.8)
        {
            cv::circle(frame, wheelStartPoint + rotatedRightPoint, 30, cv::Scalar(0, 255, 0), -1);
            wheelRightChecked = true;
        }
    }
}


cv::Mat drawAndGetRotatedPoints(cv::Mat frame, cv::Point2f wheelStartPoint, cv::Mat points, cv::Mat rotationMat)
{
    // 좌표 회전 적용
    cv::Mat rotatedPoints = points.clone();
    cv::transform(rotatedPoints, rotatedPoints, rotationMat);

    // 회전된 좌표 추출
    cv::Point2f rotatedLeftPoint(rotatedPoints.at<cv::Vec2d>(0, 0)[0], rotatedPoints.at<cv::Vec2d>(0, 0)[1]);
    cv::Point2f rotatedRightPoint(rotatedPoints.at<cv::Vec2d>(1, 0)[0], rotatedPoints.at<cv::Vec2d>(1, 0)[1]);
    cv::Point2f rotatedCenter(rotatedPoints.at<cv::Vec2d>(2, 0)[0], rotatedPoints.at<cv::Vec2d>(2, 0)[1]);


    cv::circle(frame, wheelStartPoint + rotatedLeftPoint, 30, cv::Scalar(255, 0, 0), 5);
    cv::circle(frame, wheelStartPoint + rotatedRightPoint, 30, cv::Scalar(0, 255, 0), 5);
    cv::circle(frame, wheelStartPoint + rotatedCenter, 40, cv::Scalar(255, 255, 255), 5);
    return rotatedPoints;
}
