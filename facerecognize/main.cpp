#include <iostream>
#include <opencv.hpp>		
//#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
	Mat img = imread("C:\\Users\\34630\\Desktop\\seetaface6\\facerecognize\\image\\ym2.jpg");
	imshow("ԭͼ", img);

	waitKey(0);
	return 0;
}
