#include <iostream>
#include <string>
#include <cstdio>
#include "bp.h"

using namespace std;

double sample[4][3] =
{
	{0,0,0},
	{0,1,1},
	{1,0,1},
	{1,1,0}
};

int main()
{
	Vector<Data> data;
	for (int i = 0; i < 4; i++)
	{
		Data t;
		for (int j = 0; j < 2; j++)
			t.x.push_back(sample[i][j]);
		t.y.push_back(sample[i][2]);
		data.push_back(t);
	}
	
	BP *bp = new BP();
	bp->Getdata(data);
	bp->Train();


	system("pause");
	return 0;

}
