#include <iostream>
#include <cstdio>
#include <cmath>
#include "bp.h"

using namespace std;

//��ȡѵ����������
void BP::Getdata(const Vector<Data> _data)
{
	data = _data;
}

//ѵ��
void BP::Train()
{
	cout << "Begin to train BP network!" << endl;
	InitNetwork();
	long int iters = 0;
	while (1)
	{
		for (int cnt = 0; cnt < 4; cnt++)
		{
			//��һ������ڵ㸳ֵ
			for (int i = 0; i < in_num; i++)
				x[0][i] = data.at(cnt).x[i];
				 
				ForwardTransfer();		//ǰ�����

				BackwardTransfer(cnt);	//�������
		}

		iters++;
		cout << iters << endl;
		double accu = GetAccu();
		cout << "�ܾ��ȣ�" << accu << endl;
		if (accu < Error)
		{
			cout << "All samples Accuracy is " << accu << endl;
			break;
		}
	
	}
	cout << "ѵ������,�ܵĵ�������Ϊ��" <<iters<< endl;

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
	//Ȩֵ��ֵ���ã�-0.1,0.1��֮����������ʼ��
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

//ǰ�򴫵ݹ���
void BP::ForwardTransfer()
{
	//������������ڵ�����ֵ
	for (int i = 0; i < hd_num; i++)
	{
		double t = 0;
		for (int j = 0; j < in_num; j++)
			t += weight[1][i][j] * x[0][j];		//Ȩֵ����Ӧ�ڵ�
		t += threshold[1][i];					//������ֵ
		x[1][i] = Sigmoid(t);

	}

	//�����������ڵ�����ֵ
	for (int i = 0; i < out_num; i++)
	{
		double t = 0;
		for (int j = 0; j < hd_num; j++)
			t += weight[2][i][j] * x[1][j];
		t += threshold[2][i];
		x[2][i] = Sigmoid(t);
	}
}

//���򴫲�����
void BP::BackwardTransfer(int cnt)
{
	CalcDelta(cnt);
	UpdateNetwork();
}

//���㵥�����������
double BP::GetError(int cnt)
{
	double ans = 0;
	for (int i = 0; i < out_num; i++)
		ans += 0.5*(x[2][i] - data.at(cnt).y[i])*(x[2][i] - data.at(cnt).y[i]);
	return ans;
}

//�������������ľ���
double BP::GetAccu()
{
	double ans = 0;
	//00,01,10,11��������ѭ���Ĵ�
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 2; j++)
			x[0][j] = data.at(i).x[j];	//��������
		
		ForwardTransfer();
		ans += 0.5*(x[2][0] - data.at(i).y[0])*(x[2][0] - data.at(i).y[0]);

	}
	return ans / 4;

}

//���������
void BP::CalcDelta(int cnt)
{
	//����������delta
	for (int i = 0; i < out_num; i++)
		d[2][i] = (x[2][i] - data.at(cnt).y[i])*x[2][i] * (1 - x[2][i]);
	//�����������delta
	for (int i = 0; i < hd_num; i++)
	{
		double t = 0;
		for (int j = 0; j < out_num; j++)
			t += weight[2][j][i] * d[2][j];
		d[1][i] = t * x[1][i] * (1 - x[1][i]);
	}
}

//���ݼ�������ĵ�������Ȩֵ����ֵ���е���
void BP::UpdateNetwork()
{
	//������������֮��Ȩֵ����ֵ����
	for (int i = 0; i < out_num; i++)
	{
		for (int j = 0; j < hd_num; j++)
			weight[2][i][j] -= Learn_rate * d[2][i] * x[1][j];

	}
	for (int i = 0; i < out_num; i++)
		threshold[2][i] -= Learn_rate*d[2][i];


	//������������֮���Ȩֵ����ֵ����
	for (int i = 0; i < hd_num; i++)
	{
		for (int j = 0; j < in_num; j++)
			weight[1][i][j] -= Learn_rate * d[1][i] * x[0][j];
	}
	for (int i = 0; i < hd_num; i++)
		threshold[1][i] -= Learn_rate * d[1][i];


}

//����Sigmoid����ֵ
double BP::Sigmoid(const double x)
{
	return 1 / (1 + exp(-x));
}

void BP::PrintWeight()
{
	cout << "��������Ȩֵ���£�" << endl;
	cout << "Weight[1][0][0] = " << weight[1][0][0] << endl;
	cout << "Weight[1][0][1] = " << weight[1][0][1] << endl;
	cout << "Weight[1][1][0] = " << weight[1][1][0] << endl;
	cout << "Weight[1][1][1] = " << weight[1][1][1] << endl;
	cout << "Weight[2][0][0] = " << weight[2][0][0] << endl;
	cout << "Weight[2][0][1] = " << weight[2][0][1] << endl;

	cout << "����������ֵ���£�" << endl;
	cout << "Threshold[1][0] = " << threshold[1][0] << endl;
	cout << "Threshold[1][1] = " << threshold[1][1] << endl;
	cout << "Threshold[2][0] = " << threshold[2][0] << endl;
}

