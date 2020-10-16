#include <iostream>
#include <opencv.hpp>	

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>

using namespace cv;
using namespace std;

const char *FD_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\face_detector.csta";
const char *FL_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\face_landmarker_pts5.csta";

//标识五个特征点
void mark(seeta::FaceLandmarker *fl, const SeetaImageData &image, const SeetaRect &face,Mat canvas) {
	std::vector<SeetaPointF> points = fl->mark(image, face);
	for (auto &point : points) {
		std::cout << "[" << point.x << ", " << point.y << "]" << std::endl;
		Point p;  //seetaPoit 转cvpoint
		p.x = point.x;
		p.y = point.y;
		circle(canvas, p, 3, Scalar(0, 0, 255), -1); //画出5点位置
	}
};

int main()
{
	
	int device_id = 0;

	//人脸检测构造器
	seeta::ModelSetting FD_model;
	FD_model.append(FD_path);
	FD_model.set_device(seeta::ModelSetting::CPU);
	FD_model.set_id(device_id);
	seeta::FaceDetector FD(FD_model);
	cout << "FD finished" << endl;

	//关键点构造器

	seeta::ModelSetting FL_model;
	FL_model.append(FL_path);
	FL_model.set_device(seeta::ModelSetting::CPU);
	FL_model.set_id(device_id);
	seeta::FaceLandmarker FL(FL_model);
	cout << "Facelank finished" << endl;



	Mat frame = imread("C:\\Users\\34630\\Desktop\\seetaface6\\facerecognize\\image\\ym2.jpg");
	Mat canvas;

	canvas = frame.clone();

	SeetaImageData simg;
	simg.height = frame.rows;
	simg.width = frame.cols;
	simg.channels = frame.channels();
	simg.data = frame.data;


	SeetaRect rect;
	//确定人脸位置
	auto infos = FD.detect(simg);
	int line_width = 4;
	for (int i = 0; i < infos.size; i++) {
		rect = infos.data[i].pos;
		float scale = infos.data[i].pos.width / 300.0;
		cv::rectangle(canvas,
			cv::Rect(infos.data[i].pos.x, infos.data[i].pos.y,
				infos.data[i].pos.width, infos.data[i].pos.height),
			cv::Scalar(128, 0, 0),
			scale *line_width
		);
	};

	mark(&FL,simg,rect,canvas);

	namedWindow("detected_image");
	imshow("detected_image", canvas);
	waitKey(0);

	return 0;
}
