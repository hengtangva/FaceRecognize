#include <iostream>
#include <opencv.hpp>	

#include <seeta\FaceDetector.h>

using namespace cv;
using namespace std;

const char *FD_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\face_detector.csta";


int main()
{
	//Mat img = imread("C:\\Users\\34630\\Desktop\\seetaface6\\facerecognize\\image\\ym2.jpg");
	//imshow("", img);
	//waitKey(0);
	//return 0;
	int device_id = 0;
	seeta::ModelSetting FD_model;
	FD_model.append(FD_path);
	FD_model.set_device(seeta::ModelSetting::CPU);
	FD_model.set_id(device_id);

	seeta::FaceDetector FD(FD_model);

	cout << "FD初始化成功过" << endl;

	Mat frame = imread("C:\\Users\\34630\\Desktop\\seetaface6\\facerecognize\\image\\ym2.jpg");
	Mat canvas;

	canvas = frame.clone();

	SeetaImageData simg;
	simg.height = frame.rows;
	simg.width = frame.cols;
	simg.channels = frame.channels();
	simg.data = frame.data;

	auto infos = FD.detect(simg);

	int line_width = 4;
	for (int i = 0; i < infos.size; i++) {
		float scale = infos.data[i].pos.width / 300.0;
		cv::rectangle(canvas,
			cv::Rect(infos.data[i].pos.x, infos.data[i].pos.y,
				infos.data[i].pos.width, infos.data[i].pos.height),
			cv::Scalar(128, 0, 0),
			scale *line_width
		);
	}
	namedWindow("detected_image");
	imshow("detected_image", canvas);
	waitKey(0);

	return 0;
}
