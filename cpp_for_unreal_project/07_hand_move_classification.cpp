#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
//https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_21-01.cpp

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
    // CSV 파일 읽기
    Ptr<TrainData> data = TrainData::loadFromCSV("240119_02_pts_labled_hand_dist_0.csv", 0, 25, -1);

    // 랜덤 포레스트 분류기 생성
    Ptr<RTrees> rf = RTrees::create();

    // 랜덤 포레스트 모델 학습
    rf->train(data);

    // 모델 저장
    rf->save("240119_02_pts_finger_rftree.yml");

    // 저장된 모델 로드
    Ptr<RTrees> loadedModel = Algorithm::load<RTrees>("240119_02_pts_finger_rftree.yml");

    // 분류할 데이터 준비
    Mat testSample_label4 = (Mat_<float>(1, 25) << -2.62052,2.41478, - 1.05904,0.69761, - 0.0270254, - 0.539551,0.47897, - 1.01389,0.657427, - 1.11216,0.626786 ,- 0.768131,0.57005, - 0.460746,0.489012,0.00974152,0.464182,0.273233,0.420156,0.499112,3.06481,9.61764,1.15761,8.30817,4.93629
        );
    Mat testSample_label5 = (Mat_<float>(1, 25) << 0.684344, - 0.654688,0.668225, - 0.647767,0.628475, - 0.636907,0.6035, - 0.626059,0.524778, - 0.581383,0.407533, - 0.479633,0.248506, - 0.270026, - 0.105905,0.19397, - 1.08283,1.2104, - 2.57662,2.49209,9.5697,0.380845,15.9464,0.0238828,18.5788
        );
    //Mat testSample_label3 = (Mat_<float>(1, 25) << -1.4829, -0.923632,0.757032,0.864642, -0.317196, -1.11791,1.02496,0.591135,1.22748,1.12825, -0.758276, -1.06086,0.976362,1.06391, -1.32357, -1.15633,0.749471,1.2398, -0.853373, -0.629004,1.91838,9.88122,12.3445,0.800458,0.332881);

    Mat testSample_label3 = (Mat_<float>(1, 25) << 0.736627,1.16511, -0.633472, -0.442934,0.155389,-0.0950749, -0.804645, -0.616426,0.628625,1.26245, -1.291, -0.953699,0.41388,0.447949, -1.7164, -1.77801,1.41791,1.42219,1.09309, -0.411559,3.33197,8.82719,22.8507,0.386298,1.42657);

    // 분류 예측
    float prediction = loadedModel->predict(testSample_label3);
    cout << "testSample_label3 Prediction: " << prediction << endl;
    prediction = loadedModel->predict(testSample_label4);
    cout << "testSample_label4 Prediction: " << prediction << endl;
    prediction = loadedModel->predict(testSample_label5);
    cout << "testSample_label5 Prediction: " << prediction << endl;

    return 0;
}
