#ifndef _BP_H_
#define	_BP_H_

#include <vector>
#define Learn_rate	0.6		//ѧϰ����
#define Error		3.2e-5	//�����������

#define Vector	std::vector

struct Data
{
	Vector<double> x;		//�ڵ���������
	Vector<double> y;		//�ڵ��������
};

class BP
{
public:
	void Getdata(const Vector<Data>);
	void Train();			//����ѵ��

private:
	void InitNetwork();					//���������ʼ��
	void ForwardTransfer();				//ǰ�򴫲�
	void BackwardTransfer(int);			//���򴫲�
	void CalcDelta(int);				//��������Ȩֵ�ĵ�����
	void UpdateNetwork();				//��������Ȩֵ
	double GetError(int);				//���㵥�����������
	double GetAccu();					//�������������ľ���
	double Sigmoid(const double);		//����sigmoid����ֵ
	void PrintWeight();					//��ӡ����Ȩֵ����ֵ

private:
	int in_num = 2;			//�����ڵ���
	int out_num = 1;		//�����ڵ���
	int hd_num = 2;			//������ڵ���
	
	Vector<Data> data;		//�����������
	double weight[3][2][2];	//�洢����Ȩֵ���������������
							//���洢����֮�������Ȩ����ֵΪÿ�����ڵ���

	double threshold[3][2];	//�洢�ڵ���ֵ
	double x[3][2];			//�洢����������������ֵ
	double d[3][2];			//�洢deltaѧϰ�����е�deltaֵ


};


#endif