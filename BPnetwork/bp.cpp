#include <iostream>
#include <cstdio>
#include <cmath>
#include "bp.h"

using namespace std;

//获取训练样本数据
void BP::Getdata(const Vector<Data> _data)
{
	data = _data;
}

//训练
void BP::Train()
{
	cout << "Begin to train BP network!" << endl;
	InitNetwork();
	long int iters = 0;
	while (1)
	{
		for (int cnt = 0; cnt < 4; cnt++)
		{
			//第一层输入节点赋值
			for (int i = 0; i < in_num; i++)
				x[0][i] = data.at(cnt).x[i];
				 
				ForwardTransfer();		//前向计算

				BackwardTransfer(cnt);	//反向更新
		}

		iters++;
		cout << iters << endl;
		double accu = GetAccu();
		cout << "总精度：" << accu << endl;
		if (accu < Error)
		{
			cout << "All samples Accuracy is " << accu << endl;
			break;
		}
	
	}
	cout << "训练结束,总的迭代次数为：" <<iters<< endl;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			x[0][0] = i;
			x[0][1] = j;
			ForwardTransfer();
			cout << x[2][0] << endl;
		}
	}
	
	PrintWeight();
	
}

void BP::InitNetwork()
{
	//weight[1][0][0] = 0.0543;
	//weight[1][0][1] = -0.0291;
	//weight[1][1][0] = 0.0579;
	//weight[1][1][1] = 0.0999;
	//权值阈值均用（-0.1,0.1）之间的随机数初始化
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			weight[1][i][j] = (2.0*rand() / RAND_MAX - 1.0) / 10.0;
	threshold[1][0] = (2.0*rand() / RAND_MAX - 1.0) / 10.0;
	threshold[1][1] = (2.0*rand() / RAND_MAX - 1.0) / 10.0;
	//threshold[1][0] = -0.0703;
	//threshold[1][1] = -0.0939;

	weight[2][0][0] = (2.0*rand() / RAND_MAX - 1.0) / 10.0;
	weight[2][0][1] = (2.0*rand() / RAND_MAX - 1.0) / 10.0;

	//weight[2][0][0] = 0.0801;
	//weight[2][0][1] = -0.0605;

	threshold[2][0] = (2.0*rand() / RAND_MAX - 1.0) / 10.0;
	//threshold[2][0] = -0.0109;
}

//前向传递过程
void BP::ForwardTransfer()
{
	//计算隐含层各节点的输出值
	for (int i = 0; i < hd_num; i++)
	{
		double t = 0;
		for (int j = 0; j < in_num; j++)
			t += weight[1][i][j] * x[0][j];		//权值乘相应节点
		t += threshold[1][i];					//加上阈值
		x[1][i] = Sigmoid(t);

	}

	//计算输出层各节点的输出值
	for (int i = 0; i < out_num; i++)
	{
		double t = 0;
		for (int j = 0; j < hd_num; j++)
			t += weight[2][i][j] * x[1][j];
		t += threshold[2][i];
		x[2][i] = Sigmoid(t);
	}
}

//反向传播计算
void BP::BackwardTransfer(int cnt)
{
	CalcDelta(cnt);
	UpdateNetwork();
}

//计算单个样本的误差
double BP::GetError(int cnt)
{
	double ans = 0;
	for (int i = 0; i < out_num; i++)
		ans += 0.5*(x[2][i] - data.at(cnt).y[i])*(x[2][i] - data.at(cnt).y[i]);
	return ans;
}

//计算所有样本的精度
double BP::GetAccu()
{
	double ans = 0;
	//00,01,10,11四组数据循环四次
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 2; j++)
			x[0][j] = data.at(i).x[j];	//输入数据
		
		ForwardTransfer();
		ans += 0.5*(x[2][0] - data.at(i).y[0])*(x[2][0] - data.at(i).y[0]);

	}
	return ans / 4;

}

//计算调整量
void BP::CalcDelta(int cnt)
{
	//计算输出层的delta
	for (int i = 0; i < out_num; i++)
		d[2][i] = (x[2][i] - data.at(cnt).y[i])*x[2][i] * (1 - x[2][i]);
	//计算隐含层的delta
	for (int i = 0; i < hd_num; i++)
	{
		double t = 0;
		for (int j = 0; j < out_num; j++)
			t += weight[2][j][i] * d[2][j];
		d[1][i] = t * x[1][i] * (1 - x[1][i]);
	}
}

//根据计算出来的调整量对权值和阈值进行调整
void BP::UpdateNetwork()
{
	//隐含层和输出层之间权值、阈值调整
	for (int i = 0; i < out_num; i++)
	{
		for (int j = 0; j < hd_num; j++)
			weight[2][i][j] -= Learn_rate * d[2][i] * x[1][j];

	}
	for (int i = 0; i < out_num; i++)
		threshold[2][i] -= Learn_rate*d[2][i];


	//输入层和隐含层之间的权值、阈值更新
	for (int i = 0; i < hd_num; i++)
	{
		for (int j = 0; j < in_num; j++)
			weight[1][i][j] -= Learn_rate * d[1][i] * x[0][j];
	}
	for (int i = 0; i < hd_num; i++)
		threshold[1][i] -= Learn_rate * d[1][i];


}

//计算Sigmoid函数值
double BP::Sigmoid(const double x)
{
	return 1 / (1 + exp(-x));
}

void BP::PrintWeight()
{
	cout << "网络最终权值如下：" << endl;
	cout << "Weight[1][0][0] = " << weight[1][0][0] << endl;
	cout << "Weight[1][0][1] = " << weight[1][0][1] << endl;
	cout << "Weight[1][1][0] = " << weight[1][1][0] << endl;
	cout << "Weight[1][1][1] = " << weight[1][1][1] << endl;
	cout << "Weight[2][0][0] = " << weight[2][0][0] << endl;
	cout << "Weight[2][0][1] = " << weight[2][0][1] << endl;

	cout << "网络最终阈值如下：" << endl;
	cout << "Threshold[1][0] = " << threshold[1][0] << endl;
	cout << "Threshold[1][1] = " << threshold[1][1] << endl;
	cout << "Threshold[2][0] = " << threshold[2][0] << endl;
}

