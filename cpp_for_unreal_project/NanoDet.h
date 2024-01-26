#pragma once
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <algorithm>

//https://github.com/hpc203/nanodet-opncv-dnn-cpp-python/tree/main

using namespace cv;
using namespace dnn;
using namespace std;

class NanoDet
{
public:
	NanoDet(int input_shape, float confThreshold, float nmsThreshold);
	int input_shape[2];   //// height, width

	void detect(Mat& srcimg);
	Mat resize_image(Mat srcimg, bool keep_ratio);

	std::vector<cv::Rect> get_color_filtered_boxes(cv::Mat image, cv::Mat& skin_image);
	cv::Mat get_pad_hand(cv::Mat frame, std::vector<cv::Rect> color_boxes);

	static bool compareRectByX(cv::Rect& rect1, cv::Rect& rect2) {
		return rect1.x < rect2.x;
	}
	static bool compareRectByArea(cv::Rect& rect1, cv::Rect& rect2) {
		return rect1.width * rect1.height > rect2.width * rect2.height;
	}


private:
	const int stride[3] = { 8, 16, 32 };
	const float mean[3] = { 103.53, 116.28, 123.675 };
	const float std[3] = { 57.375, 57.12, 58.395 };

	const int reg_max = 7;
	float prob_threshold;
	float iou_threshold;

	
	Scalar lower_skin = Scalar(0, 20, 70);
	Scalar upper_skin = Scalar(20, 255, 255);
	

	//Scalar lower_skin = Scalar(0, 20, 70);
	//Scalar upper_skin = Scalar(20, 190, 190);

	//const string classesFile = "coco.names";
	//vector<string> classes;

	int num_class;
	Net net;

	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
	void normalize(Mat& srcimg);
	void softmax(float* x, int length);
	void post_process(vector<Mat> outs, Mat& frame, int newh, int neww, int top, int left);
	void generate_proposal(vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes, const int stride_, Mat out_score, Mat out_box);
	const bool keep_ratio = false;

};
