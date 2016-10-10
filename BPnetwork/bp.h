#ifndef _BP_H_
#define	_BP_H_

#include <vector>
#define Learn_rate	0.6		//学习速率
#define Error		3.2e-5	//样本允许误差

#define Vector	std::vector

struct Data
{
	Vector<double> x;		//节点输入数据
	Vector<double> y;		//节点输出数据
};

class BP
{
public:
	void Getdata(const Vector<Data>);
	void Train();			//迭代训练

private:
	void InitNetwork();					//网络参数初始化
	void ForwardTransfer();				//前向传播
	void BackwardTransfer(int);			//后向传播
	void CalcDelta(int);				//计算网络权值的调整量
	void UpdateNetwork();				//更新网络权值
	double GetError(int);				//计算单个样本的误差
	double GetAccu();					//计算所有样本的精度
	double Sigmoid(const double);		//计算sigmoid函数值
	void PrintWeight();					//打印最终权值、阈值

private:
	int in_num = 2;			//输入层节点数
	int out_num = 1;		//输出层节点数
	int hd_num = 2;			//隐含层节点数
	
	Vector<Data> data;		//输入输出数据
	double weight[3][2][2];	//存储网络权值，后两个数的组合
							//来存储两层之间的连接权，数值为每层最大节点数

	double threshold[3][2];	//存储节点阈值
	double x[3][2];			//存储经激励函数后的输出值
	double d[3][2];			//存储delta学习规则中的delta值


};


#endif