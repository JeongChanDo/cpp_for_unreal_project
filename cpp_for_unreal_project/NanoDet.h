#pragma once
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <assert.h>
//https://github.com/hpc203/nanodet-opncv-dnn-cpp-python/tree/main

using namespace cv;
using namespace dnn;
using namespace std;

class NanoDet
{
public:
	NanoDet(int input_shape, float confThreshold, float nmsThreshold);
	void detect(Mat& srcimg);
	std::vector<cv::Rect> get_color_filtered_boxes(cv::Mat image, cv::Mat& skin_image);

private:
	const int stride[3] = { 8, 16, 32 };
	int input_shape[2];   //// height, width
	const float mean[3] = { 103.53, 116.28, 123.675 };
	const float std[3] = { 57.375, 57.12, 58.395 };

	const int reg_max = 7;
	float prob_threshold;
	float iou_threshold;

	Scalar lower_skin = Scalar(0, 20, 70);
	Scalar upper_skin = Scalar(20, 255, 255);

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
