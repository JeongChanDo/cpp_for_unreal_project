#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
//https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/demo.cpp



const std::map<std::string, int> str2backend{
    {"opencv", cv::dnn::DNN_BACKEND_OPENCV}, {"cuda", cv::dnn::DNN_BACKEND_CUDA}
};
const std::map<std::string, int> str2target{
    {"cpu", cv::dnn::DNN_TARGET_CPU}, {"cuda", cv::dnn::DNN_TARGET_CUDA}, {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
};

class YuNet
{
public:
    YuNet(const std::string& model_path,
        const cv::Size& input_size = cv::Size(320, 320),
        float conf_threshold = 0.6f,
        float nms_threshold = 0.3f,
        int top_k = 5000,
        int backend_id = 0,
        int target_id = 0)
        : model_path_(model_path), input_size_(input_size),
        conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
        top_k_(top_k), backend_id_(backend_id), target_id_(target_id)
    {
        model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    void setBackendAndTarget(int backend_id, int target_id)
    {
        backend_id_ = backend_id;
        target_id_ = target_id;
        model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    /* Overwrite the input size when creating the model. Size format: [Width, Height].
    */
    void setInputSize(const cv::Size& input_size)
    {
        input_size_ = input_size;
        model->setInputSize(input_size_);
    }

    cv::Mat infer(const cv::Mat image)
    {
        cv::Mat res;
        model->detect(image, res);
        return res;
    }

private:
    cv::Ptr<cv::FaceDetectorYN> model;

    std::string model_path_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    int backend_id_;
    int target_id_;
};

cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces, float fps = -1.f)
{
    static cv::Scalar box_color{ 0, 255, 0 };
    static std::vector<cv::Scalar> landmark_color{
        cv::Scalar(255,   0,   0), // right eye
        cv::Scalar(0,   0, 255), // left eye
        cv::Scalar(0, 255,   0), // nose tip
        cv::Scalar(255,   0, 255), // right mouth corner
        cv::Scalar(0, 255, 255)  // left mouth corner
    };
    static cv::Scalar text_color{ 0, 255, 0 };

    auto output_image = image.clone();

    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);

        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j)
        {
            int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)), y = static_cast<int>(faces.at<float>(i, 2 * j + 5));
            cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
        }
    }
    return output_image;
}


int main()
{
    cv::VideoCapture cap(0);  // 웹캠을 열기 위한 VideoCapture 객체 생성
    if (!cap.isOpened()) {
        std::cout << "웹캠을 열 수 없습니다." << std::endl;
        return -1;
    }


    
    cv::Mat frame;
    cv::Rect headRoi(190, 70, 260, 260);

    const int backend_id = str2backend.at("opencv");
    const int target_id = str2target.at("cpu");

    YuNet model(
        "yunet.onnx",
        cv::Size(260, 260),
        0.6,
        0.3,
        3000,
        backend_id,
        target_id
    );

    while (cap.read(frame)) 
    {
        cv::Mat headArea = frame(headRoi);
        //cv::rectangle(frame, headRoi, cv::Scalar(0, 0, 255), 1);


        model.setInputSize(headArea.size());

        auto tick_meter = cv::TickMeter();
        tick_meter.start();
        cv::Mat faces = model.infer(headArea);
        tick_meter.stop();

        /*
        // Print faces
        std::cout << cv::format("%d faces detected:\n", faces.rows);
        for (int i = 0; i < faces.rows; ++i)
        {
            int x1 = static_cast<int>(faces.at<float>(i, 0));
            int y1 = static_cast<int>(faces.at<float>(i, 1));
            int w = static_cast<int>(faces.at<float>(i, 2));
            int h = static_cast<int>(faces.at<float>(i, 3));
            float conf = faces.at<float>(i, 14);
            std::cout << cv::format("%d: x1=%d, y1=%d, w=%d, h=%d, conf=%.4f\n", i, x1, y1, w, h, conf);
        }
        */
        auto res_image = visualize(headArea, faces, (float)tick_meter.getFPS());
        // Visualize in a new window
        cv::imshow("YuNet Demo", res_image);

        cv::imshow("frame", frame);


        if (cv::waitKey(10) == 27) {  // ESC 키를 누르면 종료
            break;
        }

    }
    cap.release();
    cv::destroyAllWindows();

}