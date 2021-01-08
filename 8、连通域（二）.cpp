#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

/**********************************��һ�������г̵ı��****************************

*1���������£������������α���ͼ��

*2������ͼA��ʾ��AΪ����һ���������㣨��ʵ�ϱ��������е�һ�������İ׵㼴Ϊ�������㣩����û�б���ǹ������Aһ���µı�Ǻš�
		���Ǵ�A�����������һ���Ĺ���������������ϸ���ܣ���A���ڵ���������ȫ�����ٵ���Ȼ��ص�A�㣬����·���ϵĵ�ȫ�����ΪA�ı�š�

*3������ͼB��ʾ����������Ѿ���ǹ�����������A��,���A�����ң������ұߵĵ㶼���ΪA��ı�ţ�ֱ��������ɫ����Ϊֹ��

*4������ͼC��ʾ�����������һ���Ѿ�����ǵĵ�B�������������ĵ�(�������·�����Ϊ��ɫ�����Ҳ�����������)�����B�㿪ʼ��������������·���ϵĵ㶼����ΪB�ı�ţ���ΪB�Ѿ�����ǹ���A��ͬ���������������������������ͬ�ı�š�

*5������ͼD��ʾ������������������ϵĵ㣬��Ҳ���������ı��ȥ������Ҳ�ĵ㣬ֱ��������ɫ����Ϊֹ��

*6��������
************************************************************************************/

void bwLabel(const Mat& imgBw, Mat& imgLabeled)
{
	// ��ͼ����Χ����һ��ΪʲôҪ����һ��
	//������Ҫ��ǵ�ͼ��I
	Mat imgClone = Mat(imgBw.rows + 1, imgBw.cols + 1, imgBw.type(), Scalar(0));
	imgBw.copyTo(imgClone(Rect(1, 1, imgBw.cols, imgBw.rows)));

	//���ɱ�����ͼL
	imgLabeled.create(imgClone.size(), imgClone.type());  
	imgLabeled.setTo(Scalar::all(0));

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgClone, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	vector<int> contoursLabel(contours.size(), 0);
	int numlab = 1;
	// �����Χ����
	for (vector<vector<Point>>::size_type i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][3] >= 0) // �и�������Ҳ���Ǹ�����Ϊ������������
		{
			continue;  
		}
		for (vector<Point>::size_type k = 0; k != contours[i].size(); k++)
		{
			imgLabeled.at<uchar>(contours[i][k].y, contours[i][k].x) = numlab;
		}
		contoursLabel[i] = numlab++;
	}
	// ���������
	for (vector<vector<Point>>::size_type i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][3] < 0)  //û�и�������Ҳ����������
		{
			continue;  
		}
		for (vector<Point>::size_type k = 0; k != contours[i].size(); k++)
		{
			imgLabeled.at<uchar>(contours[i][k].y, contours[i][k].x) = contoursLabel[hierarchy[i][3]];
		}
	}
	// ��������������󣬻���Ҫ����������֮��ķ��������صı��
	for (int i = 0; i < imgLabeled.rows; i++)
	{
		for (int j = 0; j < imgLabeled.cols; j++)
		{
			if (imgClone.at<uchar>(i, j) != 0 && imgLabeled.at<uchar>(i, j) == 0)  //��I�ϲ�Ϊ0��L��δ���
			{
				imgLabeled.at<uchar>(i, j) = imgLabeled.at<uchar>(i, j - 1); //�������ֵ���
			}
		}
	}
	imgLabeled = imgLabeled(Rect(1, 1, imgBw.cols, imgBw.rows)).clone(); // ���߽�ü���1����
	cout <<"��ͨ�������"<< numlab << endl;
}

int main() {
	Mat src = imread("D:/photo/code.png");
	Mat grayImg;
	cvtColor(src, grayImg, CV_BGR2GRAY);
	Mat binImg; // I
	threshold(grayImg, binImg, 100, 255, THRESH_BINARY);
	Mat imgLabeled;  // L
	bwLabel(binImg,imgLabeled);
	imshow("",imgLabeled);

	waitKey(0);
	return 0;
}