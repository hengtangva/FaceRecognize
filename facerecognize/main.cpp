#include <iostream>
#include <opencv.hpp>	
#include <vector>
#include <memory>
#include<Windows.h>


#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>
#include <seeta/FaceAntiSpoofing.h>

using namespace cv;
using namespace std;

int device_id = 0;

const char* FaceDector_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\face_detector.csta";//人脸检测模型
const char* FaceLandmarker_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\face_landmarker_pts5.csta";//五点检测模型
const char* FaceRecognizer_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\face_recognizer.csta";//人脸特征匹配和对比模型
const char* EyeStateDetector_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\eye_state.csta";//眼睛状态检测模型
const char* fasfirst_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\fas_first.csta";//局部活体
const char* fassecond_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\fas_second.csta";//全局活体
const char* face_landmarker_mask_pts5_path = "C:\\Users\\34630\\Desktop\\seetaface6\\sf6.0_windows\\model\\sf3.0_models\\face_landmarker_mask_pts5.ctsa";

//按人脸大小排序人脸数组,提取最大人脸
void sort(SeetaFaceInfoArray face_sfia)
{
	int m = face_sfia.size;
	std::vector<SeetaFaceInfo> faces(m);
	for (int i = 0; i < face_sfia.size; i++)
	{
		faces.at(i) = face_sfia.data[i];
	}
	std::partial_sort(faces.begin(), faces.begin() + 1, faces.end(), [](SeetaFaceInfo a, SeetaFaceInfo b) {
		return a.pos.width > b.pos.width;
	});
	for (int i = 0; i < face_sfia.size; i++)
	{
		face_sfia.data[i] = faces.at(i);
	}
}

//识别人脸的位置，并用矩形标出
SeetaRect FaceRect(SeetaFaceInfoArray infos,Mat canvas) {
	SeetaRect rect;
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
	return rect;
}

//标识五个特征点
std::vector<SeetaPointF> mark (seeta::FaceLandmarker *fl, const SeetaImageData &image, const SeetaRect &face,Mat canvas) {
	std::vector<SeetaPointF> points = fl->mark(image, face);
	for (auto &point : points) {
		std::cout << "[" << point.x << ", " << point.y << "]" << std::endl;
		Point p;  //seetaPoit 转cvpoint
		p.x = point.x;
		p.y = point.y;
		circle(canvas, p, 3, Scalar(0, 0, 255), -1); //画出5点位置
	}
	return points;
};

//根据五个点提取特征
std::shared_ptr<float> extract(
	seeta::FaceRecognizer *fr,
	const SeetaImageData &image,
	const std::vector<SeetaPointF> &points) {
	std::shared_ptr<float> features(
		new float[fr->GetExtractFeatureSize()],
		std::default_delete<float[]>());
	fr->Extract(image, points.data(), features.get());
	return features;
}


////比较两个特征判断是否相似
float compare(seeta::FaceRecognizer *fr,
	const std::shared_ptr<float> &feat1,
	const std::shared_ptr<float> &feat2) {
	return fr->CalculateSimilarity(feat1.get(), feat2.get());
}



const char *SPOOF_STATE_STR[] = { "real",//真脸
                                  "spoof",//攻击脸（假脸）
                                  "fuzzy",//未识别
                                  "detecting" //检测中
};

//活体检测
int predict(seeta::FaceAntiSpoofing *fas, const SeetaImageData &image, const SeetaRect &face, std::vector<SeetaPointF> points) {
	SeetaPointF point[5];
	for (int i = 0; i < 5; i++)
	{
		point[i] = points.at(i);

	}
	auto status = fas->Predict(image, face, point);
	return status;
}

//重置识别状态
void reset_video(seeta::FaceAntiSpoofing *fas) {
	fas->ResetVideo();
}

//设置视频读取的帧数
void set_frame(int32_t number,seeta::FaceAntiSpoofing *fas)
{
	fas->SetVideoFrameCount(number);//默认是10;

}

//通用的模型构造器
seeta::ModelSetting setModel(const char* path) {
	seeta::ModelSetting model;
	model.append(path);
	model.set_device(seeta::ModelSetting::CPU);
	return model;
}

//cv图片转seetaImageData图片
SeetaImageData imgTranslate(Mat frame) {
	SeetaImageData simg;
	simg.height = frame.rows;
	simg.width = frame.cols;
	simg.channels = frame.channels();
	simg.data = frame.data;
	return simg;
};



int main(){

	
	seeta::FaceDetector FD(setModel(FaceDector_path));//人脸检测构造器
	
	seeta::FaceLandmarker FL(setModel(FaceLandmarker_path));//关键点构造器

	seeta::FaceRecognizer FR(setModel(FaceRecognizer_path)); 	// 人脸特征匹配构造

	seeta::ModelSetting model1 = setModel(fasfirst_path); //局部活体
	seeta::ModelSetting model2 = model1;
	model2.append(fassecond_path);//全局活体+局部活体
	seeta::FaceAntiSpoofing FS(model2);  

	Mat frame1 = imread("C:\\Users\\34630\\Desktop\\seetaface6\\facerecognize\\image\\me3.jpg");//2,3,5可行

	Mat canvas1;
	canvas1 = frame1.clone();

	SeetaImageData simg1 = imgTranslate(frame1);

	SeetaFaceInfoArray face1 = FD.detect(simg1);
	sort(face1);//找最大人脸

	//确定人脸位置
	SeetaRect rect1 = FaceRect(face1,canvas1);

	vector<SeetaPointF> points1 = mark(&FL, simg1, rect1, canvas1);//将得到的五点数组传给特征，以用来后面的比对

	std::shared_ptr<float> features1 = extract(&FR, simg1, points1);//提取特征点


	//对另一幅图片做同样操作
	//Mat frame2 = imread("C:\\Users\\34630\\Desktop\\seetaface6\\facerecognize\\image\\ym2.jpg");
	//Mat canvas2;
	//canvas2 = frame2.clone();
	//SeetaImageData simg2 = imgTranslate(frame2);
	//SeetaFaceInfoArray face2 = FD.detect(simg2);
	//sort(face2);
	//SeetaRect rect2 = FaceRect(face2, canvas2);
	//vector<SeetaPointF> points2 = mark(&FL, simg2, rect2, canvas2);
	//std::shared_ptr<float> features2 = extract(&FR, simg2, points2);


	//float similarity = compare(&FR,features1, features2);

	//std::cout << "the similarity1 is:  " << similarity <<endl;

	VideoCapture videocapture(0);
	Mat mat;
	while (videocapture.isOpened())
	{
		videocapture.read(mat);
		flip(mat, mat, 1);
		SeetaImageData simg0 = imgTranslate(mat);

		SeetaFaceInfoArray face0 = FD.detect(simg0);
		sort(face1);

		SeetaRect rect0 = FaceRect(face0, mat);

		vector<SeetaPointF> points0 = mark(&FL, simg0, rect0, mat);//将得到的五点数组传给特征，以用来后面的比对

		std::shared_ptr<float> features0 = extract(&FR, simg0, points0);//提取特征点

		//std::cout << compare(&FR,features0,features1) << std::endl;

		set_frame(100, &FS);
		int status = predict(&FS, simg0, rect0, points0);//获取活体检测状态
		cout << status << endl;

		imshow("1", mat);
		waitKey(1);

	}

	//namedWindow("detected_image");
	//imshow("detected_image", canvas2);
	//waitKey(0);

	return 0;
}
