#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;
using namespace cv;
/**********************************��һ�������г̵ı��****************************

* 1������ɨ��ͼ�����ǰ�ÿһ���������İ�ɫ�������һ�����г�Ϊһ����(run)���������������start�������յ�end�Լ������ڵ��кš�

* 2�����ڳ��˵�һ���������������ţ��������ǰһ���е������Ŷ�û���غ����������һ���µı�ţ������������һ����һ�������غ���������һ�е��Ǹ��ŵı�Ÿ�������
		���������һ�е�2�����ϵ������ص����������ǰ�Ÿ�һ�������ŵ���С��ţ�������һ�е��⼸���ŵı��д��ȼ۶ԣ�˵����������һ�ࡣ

* 3�����ȼ۶�ת��Ϊ�ȼ����У�ÿһ��������Ҫ��һ��ͬ�ı�ţ���Ϊ���Ƕ��ǵȼ۵ġ���1��ʼ����ÿ���ȼ�����һ����š�

* 4��������ʼ�ŵı�ǣ����ҵȼ����У����������µı�ǡ�

* 5����ÿ���ŵı��������ͼ���С�
 
* 6��������
**********************************************************/

/*
* ������fillRunVectors
* ���ܣ���������ŵĲ������¼
* ��Ӧ���裺1
*/
void fillRunVectors(const Mat& bwImage,  int& NumberOfRuns,  vector<int>& stRun,  vector<int>& enRun,  vector<int>& rowRun)
{
	for (int i = 0; i < bwImage.rows; i++)//����ɨ��
	{
		const uchar* rowData = bwImage.ptr<uchar>(i);  //ָ��� i �еĵ�һ��Ԫ��
		if (rowData[0] == 255)  //����� i �еĵ�һ��Ԫ�ػҶ�ֵΪ255
		{
			NumberOfRuns++; 
			stRun.push_back(0);
			rowRun.push_back(i);
		}
		for (int j = 1; j < bwImage.cols; j++)  //�������С� ע�Ⲣû�е����һ��Ԫ�أ���
		{
			if (rowData[j - 1] == 0 && rowData[j] == 255) //���ǰһ��Ԫ�ػҶ�ֵΪ0 ����Ԫ�ػҶ�ֵΪ255��������жϸ�Ԫ�������ŵ�ͷ
			{
				NumberOfRuns++;
				stRun.push_back(j);
				rowRun.push_back(i);
			}
			else if (rowData[j - 1] == 255 && rowData[j] == 0) //���ǰһ��Ԫ�ػҶ�ֵΪ255����Ԫ�ػҶ�ֵΪ0��������ж�ǰһ��Ԫ��Ϊ��ǰ�ŵ�β��
			{
				enRun.push_back(j - 1); 
			}
		}
		if (rowData[bwImage.cols - 1])  //�жϸ������һ��Ԫ��
		{
			enRun.push_back(bwImage.cols - 1);
		}
	}
}



/*
* ������firstPass
* ���ܣ�����ŵı����ȼ۶��б������
* ��Ӧ���裺2
*/
void firstPass(vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun, int& NumberOfRuns,vector<int>& runLabels, vector<pair<int, int>>& equivalences, int offset) //offset��4��ͨ����8��ͨ�йأ�����˵����0����1
{
	runLabels.assign(NumberOfRuns, 0); //��ʼ���ű�ǩ���������ֻ��NumberOfRuns����ǩ
	int idxLabel = 1;  //��ǩ�ţ���1��ʼ
	int curRowIdx = 0; //��ǰ���±�
	int firstRunOnCur = 0; //��ǰ�еĵ�һ���ű��
	int firstRunOnPre = 0; //��һ�е�һ���ű��
	int lastRunOnPre = -1; //��һ�����һ���ű��

	for (int i = 0; i < NumberOfRuns; i++)  //����������
	{
		// ����Ǹ��еĵ�һ��run���������һ�е�һ��run�����һ��run�����
		//rowRun�洢��ÿһ���ŵ��кţ�ʵ���ϴӵڶ����ſ�ʼ����ʵ������ 
		if (rowRun[i] != curRowIdx)
		{
			curRowIdx = rowRun[i]; // �����е����
			firstRunOnPre = firstRunOnCur;  //����һ�еĵ�һ���ŵ���ţ�����firstRunOnPre
			lastRunOnPre = i - 1;  //PS����������Կ������Ǳ������������ţ�Ҳ���ǵڼ�����
			firstRunOnCur = i;  //����firstRunOnCur����
		}
		
		for (int j = firstRunOnPre; j <= lastRunOnPre; j++)  // ������һ�е�����run���ж��Ƿ��ڵ�ǰrun���غϵ�����
		{
			// �����غ� �� �������ڵ�����
			//����stRun��enRun������������꣬����ǰ���ͨ�Ļ���offset=1
			if (stRun[i] <= enRun[j] + offset && enRun[i] >= stRun[j] - offset && rowRun[i] == rowRun[j] + 1)  
			{
				if (runLabels[i] == 0)  // û�б���Ź�
					runLabels[i] = runLabels[j]; 
				else if (runLabels[i] != runLabels[j]) // �Ѿ�����ţ���������Ǹ�������һ�������ϸ����ڵ��Ŷ����غ�  
					equivalences.push_back(make_pair(runLabels[i], runLabels[j])); // ����ȼ۶�
			}
		}
		if (runLabels[i] == 0) // û����ǰһ�е��κ�run�غ�
		{
			runLabels[i] = idxLabel++; 
		}

	}
}


/*
* ������replaceSameLabel
* ���ܣ����ȼ۶ԵĴ���Ϊ���ɸ��ȼ�����
* ��Ӧ���裺3��4
*/
void replaceSameLabel(vector<int>& runLabels, vector<pair<int, int>>&equivalence) //ÿ���ȼ۱���һ��vector<int>�����棬�ȼ۶��б�����map<pair<int,int>>��
{
	if (equivalence.size() == NULL)
	{
		return;
	}
	int maxLabel = *max_element(runLabels.begin(), runLabels.end());  //max_element����������ѯ���ֵ���ڵĵ�һ��λ�ã����صĵ�ַ��*�����÷��صľ��ǵ�ַ��ָ������ֵ
	vector<vector<bool>> eqTab(maxLabel, vector<bool>(maxLabel, false)); //��ÿһ����ǩ������һ��vector<bool>���������洢��������ǩ���Ƿ�ȼ�
	vector<pair<int, int>>::iterator vecPairIt = equivalence.begin();
	while (vecPairIt != equivalence.end())  //����ÿһ���ȼ۶�
	{
		//��Ϊ��ǩ���Ǵ�1��ʼ�ģ���vector������Ǵ�0��ʼ��
		//vecPairIt->first �ǵȼ۶��е�һ��ֵ��vecPairIt->second�ǵȼ۶��еڶ���ֵ�����ֵ�Ǳ�ǩ��
		//���磨1��2��
		eqTab[(double)vecPairIt->first - 1][(double)vecPairIt->second - 1] = true;  // eqTab[0][1]=ture Ҳ���Ǵ����� 1�ű�ǩ��2�ű�ǩ�ȼ�
		eqTab[(double)vecPairIt->second - 1][(double)vecPairIt->first - 1] = true; // eqTab[1][0]=ture Ҳ���Ǵ����� 1�ű�ǩ��2�ű�ǩ�ȼ�
		vecPairIt++; 
	}
	vector<int> labelFlag(maxLabel, 0); //�洢�ñ�ǩ���������ȼ�������
	vector<vector<int>> equaList;  // �洢���еȼ�����
	vector<int> tempList; // �����洢��ʱ�ĵȼ�����
	cout <<"maxLabel="<< maxLabel << endl;

	for (int i = 1; i <= maxLabel; i++)  // ����� i ������Ǳ�ǩ��
	{
		if (labelFlag[(double) i - 1]) //�жϸñ�ǩ�Ƿ��Ѿ������뵽���еĵȼ�������
		{
			continue; //������ǰѭ������������һѭ��
		}

		//���û�м��뵽���еȼ������У�Ҳ��������һ���µĵȼ����� �Ӷ�equaList.size() + 1��***����ԭ���� 2�������ñ�ǩ�ž��ھ��ǵ�3������***��
		 //��***Ϊ�˷�����⣬���� i = 1Ҳ����1�ű�ǩ����ʱlabelFlag[ 0 ] = 1***��
		labelFlag[(double)i - 1] = equaList.size() + 1; 

		tempList.push_back(i); // ��***�� 1�ű�ǩ������ʱ�ĵȼ�����tempList��***��

		// ������ʱ�ȼ����� ��***���ڼ��� i= 1����ʱtempList.size() = 1***��
		//��***����tempList�����ı䣬tempList.size() = 5***��
		for (vector<int>::size_type j = 0; j < tempList.size(); j++)  
		{
			
			 // ����tempList��ÿһ��ֵ��vector<bool>����������maxLabel��
			//��***j = 1ʱ �� ��ʱ tempListΪ 1 �� eqTab[0]***��
			//��***j = 2ʱ �� ��ʱ tempListΪ 1-2-3-4-5 �� eqTab[1]����������ͬ������ 2��7��8�ȼۣ���tempList��Ϊ 1-2-3-4-5-7-8�����ѭ����ֱ��������1�ȼ۵�ֵ***��
			for (vector<bool>::size_type k = 0; k != eqTab[ (double)tempList[j] - 1 ].size(); k++) 
			{
				//��***��� eqTab[0][k] = true��Ҳ���ǣ�+1�ű�ǩ��1�ű�ǩ�ȼ�***��
				//   ! labelFlag[k] = true�� ��0����Ҳ����k+1�ű�ǩ��û�д������е�������
				if (eqTab[(double)tempList[j] - 1][k] && !labelFlag[k])   
				{
					//��***���� 2��3��4��5��1�ȼۣ��� tempList��Ϊ�� 1-2-3-4-5���� labelFlag[2��3��4��5] = 1��Ҳ����˵��һ�ŵȼ�������***��
					tempList.push_back(k + 1);  // �ͽ� k+1�ű�ǩ��������ʱ�ĵȼ������У�������ʱ�������ı䣬�����forѭ��Ҳ�����ı䣡����
					labelFlag[k] = equaList.size() + 1; // Ȼ��ȼ��������Ÿ��� labelFlag��k+1�ű�ǩ��Ӧ��λ���ϵ�ֵ
				}
			}
		}

		equaList.push_back(tempList);  //������ɵĵȼ��������
		tempList.clear();  //��������ʱ��������������洢��һ���ȼ�����
	}
	cout <<"�ȼ�������Ϊ��"<< equaList.size() << endl;
	for (vector<int>::size_type i = 0; i != runLabels.size(); i++)
	{
		runLabels[i] = labelFlag[(double)runLabels[i] - 1];  //�����ŵı�ǩ�ţ�����runLabels��0��λ�ö�Ӧ��ֵ��1
	}
}


/*
* ������Drawcontours
* ���ܣ���ͼ�б����ͨ��
* ��Ӧ���裺5
*/
void Drawcontours(Mat& src, vector<int>& runLabels, vector<int>& stRun, vector<int>& enRun, vector<int>& rowRun)
{
	int RunsNumber = *max_element(runLabels.begin(), runLabels.end());// ��ͨ�����

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

		for (size_t j = 0; j < dst.cols; j++) //��������ÿһ��
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
	//ͨ�����ô��ݣ����Ըı�ʵ�ε�ֵ
	fillRunVectors(binImg,NumberOfRuns,stRun,enRun,rowRun);
	vector<int> runLabels;
	vector<pair<int, int>> equivalences;
	firstPass(stRun , enRun , rowRun , NumberOfRuns , runLabels , equivalences , 0);
	replaceSameLabel(runLabels, equivalences);
	Drawcontours(src,runLabels,stRun , enRun , rowRun);


	waitKey(0);
	return 0;
}