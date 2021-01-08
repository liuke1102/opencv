#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;
using namespace cv;
/**********************************（一）基于行程的标记****************************

* 1、逐行扫描图像，我们把每一行中连续的白色像素组成一个序列称为一个团(run)，并记下它的起点start、它的终点end以及它所在的行号。

* 2、对于除了第一行外的所有行里的团，如果它与前一行中的所有团都没有重合区域，则给它一个新的标号；如果它仅与上一行中一个团有重合区域，则将上一行的那个团的标号赋给它；
		如果它与上一行的2个以上的团有重叠区域，则给当前团赋一个相连团的最小标号，并将上一行的这几个团的标记写入等价对，说明它们属于一类。

* 3、将等价对转换为等价序列，每一个序列需要给一相同的标号，因为它们都是等价的。从1开始，给每个等价序列一个标号。

* 4、遍历开始团的标记，查找等价序列，给予它们新的标记。

* 5、将每个团的标号填入标记图像中。
 
* 6、结束。
**********************************************************/

/*
* 函数：fillRunVectors
* 功能：完成所有团的查找与记录
* 对应步骤：1
*/
void fillRunVectors(const Mat& bwImage,  int& NumberOfRuns,  vector<int>& stRun,  vector<int>& enRun,  vector<int>& rowRun)
{
	for (int i = 0; i < bwImage.rows; i++)//逐行扫描
	{
		const uchar* rowData = bwImage.ptr<uchar>(i);  //指向第 i 行的第一个元素
		if (rowData[0] == 255)  //如果第 i 行的第一个元素灰度值为255
		{
			NumberOfRuns++; 
			stRun.push_back(0);
			rowRun.push_back(i);
		}
		for (int j = 1; j < bwImage.cols; j++)  //遍历该行。 注意并没有到最后一个元素！！
		{
			if (rowData[j - 1] == 0 && rowData[j] == 255) //如果前一个元素灰度值为0 ，该元素灰度值为255，则可以判断该元素是新团的头
			{
				NumberOfRuns++;
				stRun.push_back(j);
				rowRun.push_back(i);
			}
			else if (rowData[j - 1] == 255 && rowData[j] == 0) //如果前一个元素灰度值为255，该元素灰度值为0，则可以判断前一个元素为当前团的尾部
			{
				enRun.push_back(j - 1); 
			}
		}
		if (rowData[bwImage.cols - 1])  //判断该行最后一个元素
		{
			enRun.push_back(bwImage.cols - 1);
		}
	}
}



/*
* 函数：firstPass
* 功能：完成团的标记与等价对列表的生成
* 对应步骤：2
*/
void firstPass(vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun, int& NumberOfRuns,vector<int>& runLabels, vector<pair<int, int>>& equivalences, int offset) //offset与4连通还是8联通有关，作者说不是0就是1
{
	runLabels.assign(NumberOfRuns, 0); //初始化团标签容器，最多只有NumberOfRuns个标签
	int idxLabel = 1;  //标签号，从1开始
	int curRowIdx = 0; //当前行下标
	int firstRunOnCur = 0; //当前行的第一个团标号
	int firstRunOnPre = 0; //上一行第一个团标号
	int lastRunOnPre = -1; //上一行最后一个团标号

	for (int i = 0; i < NumberOfRuns; i++)  //遍历所有团
	{
		// 如果是该行的第一个run，则更新上一行第一个run、最后一个run的序号
		//rowRun存储了每一个团的行号，实际上从第二个团开始才有实际作用 
		if (rowRun[i] != curRowIdx)
		{
			curRowIdx = rowRun[i]; // 更新行的序号
			firstRunOnPre = firstRunOnCur;  //把上一行的第一个团的序号，赋给firstRunOnPre
			lastRunOnPre = i - 1;  //PS：从这里可以看出这是变量保存的是序号（也就是第几个）
			firstRunOnCur = i;  //更新firstRunOnCur变量
		}
		
		for (int j = firstRunOnPre; j <= lastRunOnPre; j++)  // 遍历上一行的所有run，判断是否于当前run有重合的区域
		{
			// 区域重合 且 处于相邻的两行
			//这里stRun和enRun保存的是列坐标，如果是八连通的话，offset=1
			if (stRun[i] <= enRun[j] + offset && enRun[i] >= stRun[j] - offset && rowRun[i] == rowRun[j] + 1)  
			{
				if (runLabels[i] == 0)  // 没有被标号过
					runLabels[i] = runLabels[j]; 
				else if (runLabels[i] != runLabels[j]) // 已经被标号，这种情况是该团与上一行两以上个相邻的团都有重合  
					equivalences.push_back(make_pair(runLabels[i], runLabels[j])); // 保存等价对
			}
		}
		if (runLabels[i] == 0) // 没有与前一列的任何run重合
		{
			runLabels[i] = idxLabel++; 
		}

	}
}


/*
* 函数：replaceSameLabel
* 功能：将等价对的处理为若干个等价序列
* 对应步骤：3、4
*/
void replaceSameLabel(vector<int>& runLabels, vector<pair<int, int>>&equivalence) //每个等价表用一个vector<int>来保存，等价对列表保存在map<pair<int,int>>里
{
	if (equivalence.size() == NULL)
	{
		return;
	}
	int maxLabel = *max_element(runLabels.begin(), runLabels.end());  //max_element是用来来查询最大值所在的第一个位置，返回的地址，*解引用返回的就是地址所指向的最大值
	vector<vector<bool>> eqTab(maxLabel, vector<bool>(maxLabel, false)); //给每一个标签都创建一个vector<bool>容器用来存储与其他标签号是否等价
	vector<pair<int, int>>::iterator vecPairIt = equivalence.begin();
	while (vecPairIt != equivalence.end())  //遍历每一个等价对
	{
		//因为标签号是从1开始的，而vector里序号是从0开始的
		//vecPairIt->first 是等价对中第一个值，vecPairIt->second是等价对中第二个值，这个值是标签号
		//比如（1，2）
		eqTab[(double)vecPairIt->first - 1][(double)vecPairIt->second - 1] = true;  // eqTab[0][1]=ture 也就是代表了 1号标签与2号标签等价
		eqTab[(double)vecPairIt->second - 1][(double)vecPairIt->first - 1] = true; // eqTab[1][0]=ture 也就是代表了 1号标签与2号标签等价
		vecPairIt++; 
	}
	vector<int> labelFlag(maxLabel, 0); //存储该标签号在哪条等价链表中
	vector<vector<int>> equaList;  // 存储所有等价链表
	vector<int> tempList; // 用来存储临时的等价链表
	cout <<"maxLabel="<< maxLabel << endl;

	for (int i = 1; i <= maxLabel; i++)  // 这里的 i 代表的是标签号
	{
		if (labelFlag[(double) i - 1]) //判断该标签是否已经被加入到已有的等价链表中
		{
			continue; //结束当前循环，并进入下一循环
		}

		//如果没有加入到已有等价链表中，也就是生成一条新的等价链表， 从而equaList.size() + 1（***假设原来有 2条链表，该标签号就在就是第3条链表***）
		 //（***为了方便理解，假设 i = 1也就是1号标签，此时labelFlag[ 0 ] = 1***）
		labelFlag[(double)i - 1] = equaList.size() + 1; 

		tempList.push_back(i); // （***将 1号标签插入临时的等价链表tempList中***）

		// 遍历临时等价链表 （***由于假设 i= 1，此时tempList.size() = 1***）
		//（***由于tempList发生改变，tempList.size() = 5***）
		for (vector<int>::size_type j = 0; j < tempList.size(); j++)  
		{
			
			 // 遍历tempList中每一个值的vector<bool>，次数都是maxLabel次
			//（***j = 1时 ， 此时 tempList为 1 ， eqTab[0]***）
			//（***j = 2时 ， 此时 tempList为 1-2-3-4-5 ， eqTab[1]，下面流程同理，假设 2，7，8等价，则tempList变为 1-2-3-4-5-7-8，如此循环，直到所有与1等价的值***）
			for (vector<bool>::size_type k = 0; k != eqTab[ (double)tempList[j] - 1 ].size(); k++) 
			{
				//（***如果 eqTab[0][k] = true，也就是ｋ+1号标签与1号标签等价***）
				//   ! labelFlag[k] = true（ ！0），也就是k+1号标签并没有存在已有的链表中
				if (eqTab[(double)tempList[j] - 1][k] && !labelFlag[k])   
				{
					//（***假设 2，3，4，5与1等价，则 tempList变为了 1-2-3-4-5，而 labelFlag[2、3、4、5] = 1，也就是说在一号等价链表上***）
					tempList.push_back(k + 1);  // 就将 k+1号标签给加入临时的等价链表中，由于临时链表发生改变，这外层for循环也发生改变！！！
					labelFlag[k] = equaList.size() + 1; // 然后等价链表的序号赋给 labelFlag中k+1号标签对应的位置上的值
				}
			}
		}

		equaList.push_back(tempList);  //将已完成的等价链表存入
		tempList.clear();  //并将该临时链表情况，用来存储下一条等价链表
	}
	cout <<"等价链条数为："<< equaList.size() << endl;
	for (vector<int>::size_type i = 0; i != runLabels.size(); i++)
	{
		runLabels[i] = labelFlag[(double)runLabels[i] - 1];  //更新团的标签号，这里runLabels的0号位置对应的值是1
	}
}


/*
* 函数：Drawcontours
* 功能：在图中标记连通域
* 对应步骤：5
*/
void Drawcontours(Mat& src, vector<int>& runLabels, vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun)
{
	int RunsNumber = *max_element(runLabels.begin(), runLabels.end());// 连通域个数

	vector<int> Scolor;
	int color = 255 / RunsNumber;
	for (int i = 1; i <= RunsNumber; i++)
	{
		Scolor.push_back(color * i);
	}

	Mat dst = Mat::zeros(src.size(), CV_8UC1);
	for (size_t i = 0; i < rowRun.size(); i++)
	{
		uchar* dst_rowData = dst.ptr<uchar>(rowRun[i]);

		for (size_t j = 0; j < dst.cols; j++) //遍历该行每一列
		{
			if (j >= stRun[i] && j <= enRun[i]) 
			{
				dst_rowData[j] = (Scolor[(double)runLabels[i] - 1]);
			}
		}
	}
	imshow("dst", dst);
}


int main() {
	Mat src = imread("D:/photo/code.png");
	Mat grayImg;
	cvtColor(src, grayImg, CV_BGR2GRAY);
	Mat binImg;
	threshold(grayImg, binImg, 100, 255, THRESH_BINARY);
	imshow("binImg", binImg);

	int NumberOfRuns=0;
	vector<int> stRun;
	vector<int> enRun;
	vector<int> rowRun;
	//通过引用传递，可以改变实参的值
	fillRunVectors(binImg,NumberOfRuns,stRun,enRun,rowRun);
	vector<int> runLabels;
	vector<pair<int, int>> equivalences;
	firstPass(stRun , enRun , rowRun , NumberOfRuns , runLabels , equivalences , 0);
	replaceSameLabel(runLabels, equivalences);
	Drawcontours(src,runLabels,stRun , enRun , rowRun);


	waitKey(0);
	return 0;
}