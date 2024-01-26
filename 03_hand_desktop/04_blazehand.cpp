#include <opencv2/opencv.hpp>


#include <vector>

std::vector<std::array<int, 2>> HAND_CONNECTIONS = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {5, 6}, {6, 7}, {7, 8},
    {9, 10}, {10, 11}, {11, 12},
    {13, 14}, {14, 15}, {15, 16},
    {17, 18}, {18, 19}, {19, 20},
    {0, 5}, {5, 9}, {9, 13}, {13, 17}, {0, 17}
};


// var and funcs for blazepalm
struct PalmDetection {
    float ymin;
    float xmin;
    float ymax;
    float xmax;
    cv::Point2d kp_arr[7];
    float score;
};

int blazeHandSize = 256;
int blazePalmSize = 128;
float palmMinScoreThresh = 0.7;
float palmMinNMSThresh = 0.4;
int palmMinNumKeyPoints = 7;

void ResizeAndPad(
    cv::Mat& srcimg, cv::Mat& img256,
    cv::Mat& img128, float& scale, cv::Scalar& pad
);



std::vector<PalmDetection> PredictPalmDetections(cv::Mat& img, cv::Mat& frame);
PalmDetection GetPalmDetection(cv::Mat regressor, cv::Mat classificator,
    int stride, int anchor_count, int column, int row, int anchor, int offset);
float sigmoid(float x);
std::vector<PalmDetection> DenormalizePalmDetections(std::vector<PalmDetection> detections, int width, int height, cv::Scalar pad);
void DrawPalmDetections(cv::Mat& img, std::vector<PalmDetection> denormDets);
std::vector<PalmDetection> FilteringDets(std::vector<PalmDetection> detections, int width, int height);




std::vector<cv::Rect> convertHandRects(std::vector<PalmDetection> filteredDets);
void DrawRects(cv::Mat& img, std::vector<cv::Rect> rects);
void GetHandImages(cv::Mat& img, std::vector<cv::Rect> rects, std::vector<cv::Mat>& handImgs);
std::vector<cv::Mat> PredictHandDetections(std::vector<cv::Mat>& imgs);
std::vector<cv::Mat> DenormalizeHandLandmarks(std::vector<cv::Mat> imgs_landmarks, std::vector<cv::Rect> rects);
void DrawHandDetections(cv::Mat img, std::vector<cv::Mat> denorm_imgs_landmarks);


cv::dnn::Net blazePalm = cv::dnn::readNetFromONNX("c:/blazepalm_old.onnx");

cv::dnn::Net blazeHand = cv::dnn::readNetFromONNX("c:/blazehand.onnx");
float blazeHandDScale = 3;

int webcamWidth = 640;
int webcamHeight = 480;



cv::Mat img256;
cv::Mat img128;
float scale;
cv::Scalar pad;



int main() {
    cv::Mat frame;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Failed to open camera" << std::endl;
        return -1;
    }


    while (true) {
        bool ret = cap.read(frame);
        if (!ret)
        {
            break;
        }



        ResizeAndPad(frame, img256, img128, scale, pad);
        //UE_LOG(LogTemp, Log, TEXT("scale value: %f, pad value: (%f, %f)"), scale, pad[0], pad[1]);
        std::vector<PalmDetection> normDets = PredictPalmDetections(img128, frame);
        std::vector<PalmDetection> denormDets = DenormalizePalmDetections(normDets, webcamWidth, webcamHeight, pad);
        std::vector<PalmDetection> filteredDets = FilteringDets(denormDets, webcamWidth, webcamHeight);



        std::vector<cv::Rect> handRects = convertHandRects(filteredDets);
        std::vector<cv::Mat> handImgs;
        GetHandImages(frame, handRects, handImgs);
        std::vector<cv::Mat> imgs_landmarks = PredictHandDetections(handImgs);
        std::vector<cv::Mat> denorm_imgs_landmarks = DenormalizeHandLandmarks(imgs_landmarks, handRects);





        std::string dets_size_str = "filtered dets : " + std::to_string(filteredDets.size()) + ", norm dets : " + std::to_string(normDets.size()) + ", denorm dets : " + std::to_string(denormDets.size());
        cv::putText(frame, dets_size_str, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

        DrawRects(frame, handRects);
        DrawPalmDetections(frame, filteredDets);
        DrawHandDetections(frame, denorm_imgs_landmarks);


        cv::imshow("Camera Streaming", frame);
        cv::imshow("img256", img256);
        cv::imshow("img128", img128);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();

    cv::destroyAllWindows();

    return 0;
}






void ResizeAndPad(
    cv::Mat& srcimg, cv::Mat& img256,
    cv::Mat& img128, float& scale, cv::Scalar& pad
)
{
    float h1, w1;
    int padw, padh;

    cv::Size size0 = srcimg.size();
    if (size0.height >= size0.width) {
        h1 = 256;
        w1 = 256 * size0.width / size0.height;
        padh = 0;
        padw = static_cast<int>(256 - w1);
        scale = size0.width / static_cast<float>(w1);
    }
    else {
        h1 = 256 * size0.height / size0.width;
        w1 = 256;
        padh = static_cast<int>(256 - h1);
        padw = 0;
        scale = size0.height / static_cast<float>(h1);
    }

    //UE_LOG(LogTemp, Log, TEXT("scale value: %f, size0.height: %d, size0.width : %d, h1 : %f"), scale, size0.height, size0.width, h1);

    int padh1 = static_cast<int>(padh / 2);
    int padh2 = static_cast<int>(padh / 2) + static_cast<int>(padh % 2);
    int padw1 = static_cast<int>(padw / 2);
    int padw2 = static_cast<int>(padw / 2) + static_cast<int>(padw % 2);
    pad = cv::Scalar(static_cast<float>(padh1 * scale), static_cast<float>(padw1 * scale));


    cv::resize(srcimg, img256, cv::Size(w1, h1));
    cv::copyMakeBorder(img256, img256, padh1, padh2, padw1, padw2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::resize(img256, img128, cv::Size(128, 128));
    cv::cvtColor(img256, img256, cv::COLOR_BGR2RGB);
    cv::cvtColor(img128, img128, cv::COLOR_BGR2RGB);
}






std::vector<PalmDetection> PredictPalmDetections(cv::Mat& img, cv::Mat& frame)
{
    std::vector<PalmDetection> beforeNMSResults;
    std::vector<PalmDetection> afterNMSResults;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<cv::Rect> boundingBoxes;


    cv::Mat tensor;
    img.convertTo(tensor, CV_32F);
    tensor = tensor / 127.5 - 1;
    cv::Mat blob = cv::dnn::blobFromImage(tensor, 1.0, tensor.size(), 0, false, false, CV_32F);
    std::vector<cv::String> outNames(2);
    outNames[0] = "regressors";
    outNames[1] = "classificators";

    blazePalm.setInput(blob);
    std::vector<cv::Mat> outputs;
    blazePalm.forward(outputs, outNames);

    cv::Mat classificator = outputs[0];
    cv::Mat regressor = outputs[1];


    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            for (int a = 0; a < 2; ++a) {
                PalmDetection res = GetPalmDetection(regressor, classificator, 8, 2, x, y, a, 0);
                if (res.score != 0)
                {
                    beforeNMSResults.push_back(res);

                    cv::Point2d startPt = cv::Point2d(res.xmin * 128, res.ymin * 128);
                    cv::Point2d endPt = cv::Point2d(res.xmax * 128, res.ymax * 128);

                    boundingBoxes.push_back(cv::Rect(startPt, endPt));
                    scores.push_back(res.score);
                }
            }
        }
    }

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            for (int a = 0; a < 6; ++a) {
                PalmDetection res = GetPalmDetection(regressor, classificator, 16, 6, x, y, a, 512);
                if (res.score != 0)
                {
                    beforeNMSResults.push_back(res);
                    cv::Point2d startPt = cv::Point2d(res.xmin * 128, res.ymin * 128);
                    cv::Point2d endPt = cv::Point2d(res.xmax * 128, res.ymax * 128);

                    boundingBoxes.push_back(cv::Rect(startPt, endPt));
                    scores.push_back(res.score);
                }
            }
        }
    }






    std::vector<PalmDetection> denormBeforeNMSResults = DenormalizePalmDetections(beforeNMSResults, 640, 480, cv::Scalar(80, 0));


    for (auto& denormBeforeNMSResult : denormBeforeNMSResults)
    {


        cv::Point2d startPt = cv::Point2d(denormBeforeNMSResult.xmin, denormBeforeNMSResult.ymin);
        cv::Point2d endPt = cv::Point2d(denormBeforeNMSResult.xmax, denormBeforeNMSResult.ymax);
        //cv::rectangle(frame, cv::Rect(startPt, endPt), cv::Scalar(0, 0, 255), 1);


        std::string score_str = std::to_string(static_cast<int>(denormBeforeNMSResult.score * 100));
        //cv::putText(frame, score_str, cv::Point(startPt.x, startPt.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 2);


        /*
        for (int i = 0; i < palmMinNumKeyPoints; i++)
            cv::circle(frame, denormBeforeNMSResult.kp_arr[i], 3, cv::Scalar(0, 0, 255), -1);
        */
    }


    cv::dnn::NMSBoxes(boundingBoxes, scores, palmMinScoreThresh, palmMinNMSThresh, indices);



    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        afterNMSResults.push_back(beforeNMSResults[idx]);
    }










    std::vector<PalmDetection> denormAfterNMSResults = DenormalizePalmDetections(afterNMSResults, 640, 480, cv::Scalar(80, 0));


    for (auto& denormAfterNMSResult : denormAfterNMSResults)
    {


        cv::Point2d startPt = cv::Point2d(denormAfterNMSResult.xmin, denormAfterNMSResult.ymin);
        cv::Point2d endPt = cv::Point2d(denormAfterNMSResult.xmax, denormAfterNMSResult.ymax);
        cv::rectangle(frame, cv::Rect(startPt, endPt), cv::Scalar(0, 255, 255), 2);


        for (int i = 0; i < palmMinNumKeyPoints; i++)
            cv::circle(frame, denormAfterNMSResult.kp_arr[i], 6, cv::Scalar(0, 255, 255), -1);
    }












    return afterNMSResults;
}

PalmDetection GetPalmDetection(cv::Mat regressor, cv::Mat classificator,
    int stride, int anchor_count, int column, int row, int anchor, int offset) {

    PalmDetection res;

    int index = (int(row * 128 / stride) + column) * anchor_count + anchor + offset;
    float origin_score = regressor.at<float>(0, index, 0);
    float score = sigmoid(origin_score);
    if (score < palmMinScoreThresh) return res;

    float x = classificator.at<float>(0, index, 0);
    float y = classificator.at<float>(0, index, 1);
    float w = classificator.at<float>(0, index, 2);
    float h = classificator.at<float>(0, index, 3);

    x += (column + 0.5) * stride - w / 2;
    y += (row + 0.5) * stride - h / 2;

    res.ymin = (y) / blazePalmSize;
    res.xmin = (x) / blazePalmSize;
    res.ymax = (y + h) / blazePalmSize;
    res.xmax = (x + w) / blazePalmSize;

    //std::cout << "score : " << score << ", coord : " << res.ymin << ", " << res.xmin << ", " << res.ymax << ", " << res.xmax << std::endl;
    //if ((res.ymin < 0) || (res.xmin < 0) || (res.xmax > 1) || (res.ymax > 1)) return res;

    res.score = score;

    std::vector<cv::Point2d> kpts;
    for (int key_id = 0; key_id < palmMinNumKeyPoints; key_id++)
    {
        float kpt_x = classificator.at<float>(0, index, 4 + key_id * 2);
        float kpt_y = classificator.at<float>(0, index, 5 + key_id * 2);
        kpt_x += (column + 0.5) * stride;
        kpt_x = kpt_x / blazePalmSize;
        kpt_y += (row + 0.5) * stride;
        kpt_y = kpt_y / blazePalmSize;
        //UE_LOG(LogTemp, Log, TEXT("kpt id(%d) : (%f, %f)"), key_id, kpt_x, kpt_y);
        res.kp_arr[key_id] = cv::Point2d(kpt_x, kpt_y);

    }
    return res;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}


std::vector<PalmDetection> DenormalizePalmDetections(std::vector<PalmDetection> detections, int width, int height, cv::Scalar pad)
{

    std::vector<PalmDetection> denormDets;

    int scale = 0;
    if (width > height)
        scale = width;
    else
        scale = height;

    for (auto& det : detections)
    {
        PalmDetection denormDet;
        denormDet.ymin = det.ymin * scale - pad[0];
        denormDet.xmin = det.xmin * scale - pad[1];
        denormDet.ymax = det.ymax * scale - pad[0];
        denormDet.xmax = det.xmax * scale - pad[1];
        denormDet.score = det.score;

        for (int i = 0; i < palmMinNumKeyPoints; i++)
        {
            cv::Point2d pt_new = cv::Point2d(det.kp_arr[i].x * scale - pad[1], det.kp_arr[i].y * scale - pad[0]);
            //UE_LOG(LogTemp, Log, TEXT("denorm kpt id(%d) : (%f, %f)"), i, pt_new.x, pt_new.y);
            denormDet.kp_arr[i] = pt_new;
        }
        denormDets.push_back(denormDet);
    }
    return denormDets;
}

void DrawPalmDetections(cv::Mat& img, std::vector<PalmDetection> denormDets)
{
    for (auto& denormDet : denormDets)
    {
        cv::Point2d startPt = cv::Point2d(denormDet.xmin, denormDet.ymin);
        cv::Point2d endPt = cv::Point2d(denormDet.xmax, denormDet.ymax);
        cv::rectangle(img, cv::Rect(startPt, endPt), cv::Scalar(255, 0, 0), 1);

        for (int i = 0; i < palmMinNumKeyPoints; i++)
            cv::circle(img, denormDet.kp_arr[i], 5, cv::Scalar(255, 0, 0), -1);

        std::string score_str = std::to_string(static_cast<int>(denormDet.score * 100)) + "%";
        cv::putText(img, score_str, cv::Point(startPt.x, startPt.y - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

    }
}


std::vector<PalmDetection> FilteringDets(std::vector<PalmDetection> detections, int width, int height)
{
    std::vector<PalmDetection> filteredDets;

    for (auto& denormDet : detections)
    {
        cv::Point2d startPt = cv::Point2d(denormDet.xmin, denormDet.ymin);
        if (startPt.x < 10 || startPt.y < 10)
            continue;
        if (startPt.x > width || startPt.y > height)
            continue;
        int w = denormDet.xmax - denormDet.xmin;
        int y = denormDet.ymax - denormDet.ymin;
        if ((w * y < 40 * 40) || (w * y > (width * 0.7) * (height * 0.7)))
            continue;
        filteredDets.push_back(denormDet);
    }
    return filteredDets;
}








std::vector<cv::Rect> convertHandRects(std::vector<PalmDetection> filteredDets)
{
    std::vector<cv::Rect> handRects;
    for (auto& det : filteredDets)
    {
        int width = det.xmax - det.xmin; 
        int height = det.ymax - det.ymin;
        int large_length = 0;

        int center_x = static_cast<int>((det.xmax + det.xmin) / 2);
        int center_y = static_cast<int>((det.ymax + det.ymin) / 2);

        if (width > height)
            large_length = height * blazeHandDScale;
        else
            large_length = height * blazeHandDScale;

        int start_x = center_x - static_cast<int>(large_length / 2);
        int start_y = center_y - static_cast<int>(large_length / 2) - large_length * 0.1;

        int end_x = center_x + static_cast<int>(large_length / 2);
        int end_y = center_y + static_cast<int>(large_length / 2) - large_length * 0.1;


        start_x = std::max(start_x, 0);
        start_y = std::max(start_y, 0);
        end_x = std::min(end_x, webcamWidth);
        end_y = std::min(end_y, webcamHeight);

        cv::Point2d startPt = cv::Point2d(start_x, start_y);
        cv::Point2d endPt = cv::Point2d(end_x, end_y);

        cv::Rect handRect(startPt, endPt);
        handRects.push_back(handRect);
    }

    return handRects;
}

void DrawRects(cv::Mat& img, std::vector<cv::Rect> rects)
{
    for (auto& rect : rects)
    {
        cv::rectangle(img, rect, cv::Scalar(255, 255, 255), 1);
    }
}

void GetHandImages(cv::Mat& img, std::vector<cv::Rect> rects, std::vector<cv::Mat>& handImgs)
{
    for (auto& rect : rects)
    {

        //std::cout << "img size : " << img.size() << ", rect : " << rect << std::endl;

        //cv::Mat roi(cv::Size(rect.width, rect.height), img.type(), cv::Scalar(0, 0, 0))

        cv::Mat roi = img(rect).clone();
        cv::cvtColor(roi, roi, cv::COLOR_BGR2RGB);
        cv::resize(roi, roi, cv::Size(blazeHandSize, blazeHandSize));
        handImgs.push_back(roi);
    }
}



std::vector<cv::Mat> PredictHandDetections(std::vector<cv::Mat>& imgs)
{
    std::vector<cv::Mat> imgs_landmarks;


    for (auto& img : imgs)
    {
        cv::Mat tensor;
        img.convertTo(tensor, CV_32F);
        tensor = tensor / blazeHandSize;
        cv::Mat blob = cv::dnn::blobFromImage(tensor, 1.0, tensor.size(), 0, false, false, CV_32F);
        std::vector<cv::String> outNames(3);
        outNames[0] = "hand_flag";
        outNames[1] = "handedness";
        outNames[2] = "landmarks";



        blazeHand.setInput(blob);
        std::vector<cv::Mat> outputs;
        blazeHand.forward(outputs, outNames);

        cv::Mat handflag = outputs[0];
        cv::Mat handedness = outputs[1];
        cv::Mat landmarks = outputs[2];


        imgs_landmarks.push_back(landmarks);
    }
    return imgs_landmarks;
}

std::vector<cv::Mat> DenormalizeHandLandmarks(std::vector<cv::Mat> imgs_landmarks, std::vector<cv::Rect> rects)
{
    std::vector<cv::Mat> denorm_imgs_landmarks;

    for (int i = 0; i < imgs_landmarks.size(); i++)
    {
        cv::Mat landmarks = imgs_landmarks.at(i);
        cv::Rect rect = rects.at(i);
        cv::Mat squeezed = landmarks.reshape(0, landmarks.size[1]);

        //std::cout << "squeezed size: " << squeezed.size[0] << "," << squeezed.size[1] << std::endl;

        for (int j = 0; j < squeezed.size[0]; j++)
        {
            squeezed.at<float>(j, 0) = squeezed.at<float>(j, 0) * rect.width + rect.x;
            squeezed.at<float>(j, 1) = squeezed.at<float>(j, 1) * rect.height + rect.y;
            squeezed.at<float>(j, 2) = squeezed.at<float>(j, 2) * rect.height * -1;


        }
        denorm_imgs_landmarks.push_back(squeezed);
    }

    //std::cout << std::endl;
    return denorm_imgs_landmarks;
}

void DrawHandDetections(cv::Mat img, std::vector<cv::Mat> denorm_imgs_landmarks)
{
    for (auto& denorm_landmarks : denorm_imgs_landmarks)
    {
        for (int i = 0; i < denorm_landmarks.size[0]; i++)
        {
            float x = denorm_landmarks.at<float>(i, 0);
            float y = denorm_landmarks.at<float>(i, 1);

            cv::Point pt(x, y);

            cv::circle(img, pt, 5, cv::Scalar(0, 255, 0), -1);

        }

        for (auto& connection : HAND_CONNECTIONS)
        {
            int startPtIdx = connection[0];
            int endPtIdx = connection[1];

            float startPtX = denorm_landmarks.at<float>(startPtIdx, 0);
            float startPtY = denorm_landmarks.at<float>(startPtIdx, 1);
            float endPtX = denorm_landmarks.at<float>(endPtIdx, 0);
            float endPtY = denorm_landmarks.at<float>(endPtIdx, 1);
            cv::line(img, cv::Point(startPtX, startPtY), cv::Point(endPtX, endPtY), cv::Scalar(255, 255, 255), 3);


        }
    }
}