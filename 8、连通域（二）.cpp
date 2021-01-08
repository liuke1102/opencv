#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

/**********************************（一）基于行程的标记****************************

*1、从上至下，从左至右依次遍历图像。

*2、如下图A所示，A为遇到一个外轮廓点（其实上遍历过程中第一个遇到的白点即为外轮廓点），且没有被标记过，则给A一个新的标记号。
		我们从A点出发，按照一定的规则（这个规则后面详细介绍）将A所在的外轮廓点全部跟踪到，然后回到A点，并将路径上的点全部标记为A的标号。

*3、如下图B所示，如果遇到已经标记过的外轮廓点A′,则从A′向右，将它右边的点都标记为A′的标号，直到遇到黑色像素为止。

*4、如下图C所示，如果遇到了一个已经被标记的点B，且是内轮廓的点(它的正下方像素为黑色像素且不在外轮廓上)，则从B点开始，跟踪内轮廓，路径上的点都设置为B的标号，因为B已经被标记过与A相同，所以内轮廓与外轮廓将标记相同的标号。

*5、如下图D所示，如果遍历到内轮廓上的点，则也是用轮廓的标号去标记它右侧的点，直到遇到黑色像素为止。

*6、结束。
************************************************************************************/

void bwLabel(const Mat& imgBw, Mat& imgLabeled)
{
	// 对图像周围扩充一格，为什么要扩充一格？
	//生成需要标记的图像I
	Mat imgClone = Mat(imgBw.rows + 1, imgBw.cols + 1, imgBw.type(), Scalar(0));
	imgBw.copyTo(imgClone(Rect(1, 1, imgBw.cols, imgBw.rows)));

	//生成保存标记图L
	imgLabeled.create(imgClone.size(), imgClone.type());  
	imgLabeled.setTo(Scalar::all(0));

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgClone, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	vector<int> contoursLabel(contours.size(), 0);
	int numlab = 1;
	// 标记外围轮廓
	for (vector<vector<Point>>::size_type i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][3] >= 0) // 有父轮廓，也就是该轮廓为内轮廓，跳出
		{
			continue;  
		}
		for (vector<Point>::size_type k = 0; k != contours[i].size(); k++)
		{
			imgLabeled.at<uchar>(contours[i][k].y, contours[i][k].x) = numlab;
		}
		contoursLabel[i] = numlab++;
	}
	// 标记内轮廓
	for (vector<vector<Point>>::size_type i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][3] < 0)  //没有父轮廓，也就是外轮廓
		{
			continue;  
		}
		for (vector<Point>::size_type k = 0; k != contours[i].size(); k++)
		{
			imgLabeled.at<uchar>(contours[i][k].y, contours[i][k].x) = contoursLabel[hierarchy[i][3]];
		}
	}
	// 标记完内外轮廓后，还需要将内外轮廓之间的非轮廓像素的标记
	for (int i = 0; i < imgLabeled.rows; i++)
	{
		for (int j = 0; j < imgLabeled.cols; j++)
		{
			if (imgClone.at<uchar>(i, j) != 0 && imgLabeled.at<uchar>(i, j) == 0)  //在I上不为0，L上未标记
			{
				imgLabeled.at<uchar>(i, j) = imgLabeled.at<uchar>(i, j - 1); //用左侧标记值标记
			}
		}
	}
	imgLabeled = imgLabeled(Rect(1, 1, imgBw.cols, imgBw.rows)).clone(); // 将边界裁剪掉1像素
	cout <<"连通域个数："<< numlab << endl;
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