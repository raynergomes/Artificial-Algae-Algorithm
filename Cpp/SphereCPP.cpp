#include "pch.h"
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h> 


int MaxFEVs = 1000;   // Maximum Fitness Calculation Number
const int N = 200;           // Number of Algal Colony in Population
const int D = 150;           // Number of Dimension in Problem
int LB = -100;        // Minimum Values for each dimension(ones(1, Parameters.D) *(-100);)
int UB = 100;        // Maximum Values for each dimension(ones(1, Parameters.D) * 100;)
int K = 2;           // Shear Force
double le = 0.3;         // Energy Loss
double Ap = 0.5;        // Adaptation
const int Nr = 5;  //No. of runs
#define PI 3.141592653589793
class Sphere{
public:
	void Multiplication(double x[N][D], double mul[][D]) {
		int i, j;
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < D; j++)
			{
				mul[i][j] = x[i][j] * x[i][j];
			}
		}
	}

	void transpose(double x[N][D]) {
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < i; j++)
			{
				if (i != j) {
					double temp = x[i][j];
					x[i][j] = x[j][i];
					x[j][i] = temp;
				}
			}
		}
	}
	double * sum(double x[][D]) {
		int total = 0;
		double objValue[N];
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < D; j++)
			{
				total += x[i][j];
			}
			objValue[i] = total;
			total = 0;
		}
		return objValue;
	}
	double * ObjVal(double colony[][D]) {

		double b[N][D];
		double objValue[N];
		Multiplication(colony, b);
		double* a = sum(b);
		for (int i = 0; i < N; i++)
			objValue[i] = *(a + i);
		return objValue;

	}
};
	
class CalculateGreatness {
public:
	void CalculateGreatness_(double Big_Algae[N],double Obj_Algae[N]) {
		double ObjX[N];
		for (int i = 0; i < N; i++)
		{
			ObjX[i] = (Obj_Algae[i] - max_min(Obj_Algae, 0)) / ptp(Obj_Algae);
		}
		//s2 = N
		for (int i = 0; i < N; i++)
		{
			double fKs = abs(Big_Algae[i] / 2.0);
			
			double M = (ObjX[i] / (fKs + ObjX[i]));

			double dX = M * Big_Algae[i];

			Big_Algae[i] = Big_Algae[i] + dX;
		}
	}
	double max_min(double Obj_Algae[N],int x) {
		double minElement = Obj_Algae[0];
		double maxElement = Obj_Algae[0];

		for (int i = 0; i < N; i++) {
			if (Obj_Algae[i] < minElement)
				minElement = Obj_Algae[i];
			if (Obj_Algae[i] > maxElement)
				maxElement = Obj_Algae[i];
		}
		if(x==1)
			return maxElement;
		else
			return minElement;
	}
	double ptp(double Obj_Algae[N]) {
		return max_min(Obj_Algae,1) - max_min(Obj_Algae,0);
	}
};

class GreatnessOrder {
public:
	//s2=N
	CalculateGreatness CalculateGreatness;
	double * GreatnessOrder_(double Big_Algae[N]) {
		double sorting[N];
		double BAlgae_Great_Surface[N];
		int i_sort, j_sort, l_sort, k_sort;
		for (int i = 0; i < N; i++)
		{
			BAlgae_Great_Surface[i] = 0;
			sorting[i] = i;
		}
		int i;
		for (i = 0; i < N-1; i++)
		{
			for (int j = i+1; j < N; j++)
			{
				i_sort = (int)(sorting[i]);
				j_sort = (int)(sorting[j]);
				if (Big_Algae[i_sort] > Big_Algae[j_sort]) {
					int temp = i_sort;//(int)(sorting[i]);
					sorting[i] = sorting[j];
					sorting[j] = temp;
				}
			}
			l_sort = (int)(sorting[i]);
			BAlgae_Great_Surface[l_sort] = pow(i, 2);
		}
		k_sort = (int)(sorting[N-2]);
		BAlgae_Great_Surface[k_sort] = pow(i, 2);
		double copySurface[N];
		for (int i = 0; i < N; i++)
		{
			copySurface[i] = BAlgae_Great_Surface[i];
		}
		for (int i = 0; i < N; i++)
		{

			BAlgae_Great_Surface[i] = (copySurface[i] - CalculateGreatness.max_min(copySurface, 0)) / CalculateGreatness.ptp(copySurface);
		}
		return BAlgae_Great_Surface;
	}
};

class FrictionSurface {
public:
	//s2=N
	CalculateGreatness CalculateGreatness;
	double * FrictionSurface_(double Big_Algae[N]) {
		double BAlgae_Fr_Surface[N];
		double r;
		for (int i = 0; i < N; i++)
		{
			BAlgae_Fr_Surface[i] = 0;
		}
		for (int i = 0; i < N; i++)
		{
			r = pow(((Big_Algae[i]*3) / (4*PI)), 0.333333); // Calculate the Radius
			double power = pow(r, 2);
			BAlgae_Fr_Surface[i] = 2 * PI * power;// Calculate the Friction Surface
			//printf("%f ", BAlgae_Fr_Surface[i]);
		}
		double copySurface[N];
		for (int i = 0; i < N; i++)
		{
			copySurface[i] = BAlgae_Fr_Surface[i];
		}
		for (int i = 0; i < N; i++)
		{
			//printf("%f ", BAlgae_Fr_Surface[i]);
			BAlgae_Fr_Surface[i] = (copySurface[i] - CalculateGreatness.max_min(copySurface, 0)) / CalculateGreatness.ptp(copySurface);
			//printf("%f ", BAlgae_Fr_Surface[i]);
		}
		return BAlgae_Fr_Surface;
	}
};
void write1(double x[N]) {
	for (int i = 0; i < N; i++)
	{
		printf("%f ", x[i]);
	}
}
void write2(double x[N][D]) {
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < D; j++)
		{
			printf("%f ", x[i][j]);
		}
		printf("\n");
	}
}

int * randomPermutationN(int x) {
	int index;
	int random[N];
	
	for (int i = 0; i < x; i++)
	{
		random[i] = i;
	}
	double temp;
	for (int i = 0; i < x; i++)
	{
		index = rand() % x;
		temp = random[i];
		random[i] = random[index];
		random[index] = temp;
	}
	return random;
}
int * randomPermutationD(int x) {
	int index;
	int random[D];

	for (int i = 0; i < x; i++)
	{
		random[i] = i;
	}
	double temp;
	for (int i = 0; i < x; i++)
	{
		int  v = rand();
		index = rand() % x;
		temp = random[i];
		random[i] = random[index];
		random[index] = temp;
	}
	return random;
}

class Tournement_selection {
public:
	int Tournement_selection_(double Obj_Algae[N]) {
		//s2=N
		double choice;
		int neighbor[N];
		int * x = randomPermutationN(N);
		for (int i = 0; i < N; i++)
			neighbor[i] = *(x + i);
		

		if (Obj_Algae[neighbor[0]] < Obj_Algae[neighbor[1]])
			choice = neighbor[0];
		else
			choice = neighbor[1];
		return choice;
	}
};

class AAA {
public:
	double AAA_() {
		int Neighbor;
		double Algae[N][D];
		double Obj_Algae[N];
		double Starve[N];
		double Big_Algae[N];
		double Best_Algae[D];
		double Cloro_ALG[N];
		double Big_Algae_Surface[N];
		double New_Algae[1][D];
		double Obj_New_Algae[N];

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < D; j++)
			{
				double random = (rand() % 100000) / 100000.0;
				Algae[i][j] = LB + (UB - LB) * random;
			}
			Starve[i] = 0;
			Big_Algae[i]=1;
		}

		Sphere sphere;
		CalculateGreatness CalculateGreatness;
		GreatnessOrder GreatnessOrder;
		FrictionSurface FrictionSurface;
		Tournement_selection Tournement_selection;
		int parameters[N];
		double * x = sphere.ObjVal(Algae);
		for (int i = 0; i < N; i++)
			Obj_Algae[i] = *(x + i);
		double min_Obj_Alg = CalculateGreatness.max_min(Obj_Algae, 0);//np.min
		double value;
		double Obj_Best_Algae;
		int indices;
		for (int i = 0; i < N; i++)
		{
			if (min_Obj_Alg == Obj_Algae[i]) {
				indices = i;
				Obj_Best_Algae = min_Obj_Alg;
			}
		}
		
		for (int i = 0; i < D; i++)
		{
			Best_Algae[i] = Algae[indices][i];//Best_Algae = Algae[indices, :]
		}
		CalculateGreatness.CalculateGreatness_(Big_Algae, Obj_Algae);
		int counter = 0;
		int c = N,i,starve;

		while(c < MaxFEVs){
			
			double * y = GreatnessOrder.GreatnessOrder_(Big_Algae); //Calculate energy values
			for (int m = 0; m < N; m++)
				Cloro_ALG[m] = *(y + m);
			double * z = FrictionSurface.FrictionSurface_(Big_Algae); //Sorting by descending size and normalize between[0, 1]
			for (int n = 0; n < N; n++)
				Big_Algae_Surface[n] = *(z + n);
			
			for (i = 0; i < N; i++)
			{
				starve = 0;
				while (Cloro_ALG[D-1] >= 0 && c < MaxFEVs) {
					Neighbor = Tournement_selection.Tournement_selection_(Obj_Algae);
					while (Neighbor == i) {
						Neighbor = Tournement_selection.Tournement_selection_(Obj_Algae);
					}
					int * t = randomPermutationD(D);
					for (int r = 0; r < N; r++)
						parameters[r] = *(t + r);
					for (int j = 0; j < D; j++)//New_Algae = Algae[i, :]
					{
						New_Algae[0][j] = Algae[i][j];
					}
					int parameter0 = ((int)(parameters[0]));
					int parameter1 = ((int)(parameters[1]));
					int parameter2 = ((int)(parameters[2]));

					float Subtr_Eq0 = ((float)(Algae[Neighbor][parameter0] - New_Algae[0][parameter0]));
					float Subtr_Eq1 = ((float)(Algae[Neighbor][parameter0] - New_Algae[0][parameter0]));
					float Subtr_Eq2 = ((float)(Algae[Neighbor][parameter0] - New_Algae[0][parameter0]));

					float K_Big_Algae = K - ((float)(Big_Algae_Surface[i]));
					double random = (rand() % 100000) / 100000.0;
					double rand_value = random - 0.5;
					random = (rand() % 100000) / 100000.0;
					double cosine_value = cos(random * 360);
					random = (rand() % 100000) / 100000.0;
					double sine_value = sin(random * 360);
					New_Algae[0][parameter0] = Subtr_Eq0 * K_Big_Algae * (rand_value * 2);
					New_Algae[0][parameter1] = Subtr_Eq1 * K_Big_Algae * cosine_value;
					New_Algae[0][parameter2] = Subtr_Eq2 * K_Big_Algae * sine_value;
					/////////////////////////////////////////////////////
					for (int p = 1; p < 3; p++)
					{
						if (New_Algae[0][parameters[p]] > UB)
							New_Algae[0][parameters[p]] = UB;

						if (New_Algae[0][parameters[p]] < LB)
							New_Algae[0][parameters[p]] = LB;
					}
					
					double * x = sphere.ObjVal(New_Algae);
					for (int e = 0; e < N; e++)
						Obj_New_Algae[e] = *(x + e);
					c = c + 1;
					counter = c;
					Cloro_ALG[i] = Cloro_ALG[i] - (le / 2);
					if (Obj_New_Algae[0] <= Obj_Algae[i]) {
						for (int k = 0; k < D; k++)
							Algae[i][k] = New_Algae[0][k];
						Obj_Algae[i] = Obj_New_Algae[0];
						starve = 1;
					}
					else
						Cloro_ALG[i] = Cloro_ALG[i] - (le / 2);
				}
			}
			if (starve == 0) {
				Starve[i-1] = 1;
			}
			//[val, ind] = np.min(Obj_Algae)
			double min_Obj_Alg1 = CalculateGreatness.max_min(Obj_Algae, 0);
			double valki;
			int ind;
	
			for (int i = 0; i < N; i++)
			{
				if (min_Obj_Alg == Obj_Algae[i]) {
					ind = i;
					valki = min_Obj_Alg;
				};
			}
			if (valki < Obj_Best_Algae) {
				for (int i = 0; i < D; i++)
				{
					Best_Algae[i] = Algae[ind][i];
				}
				Obj_Best_Algae = valki;
			}
			CalculateGreatness.CalculateGreatness_(Big_Algae, Obj_Algae);
			double random = (rand() % 100000) / 100000.0;
			int m = ((int)(random * D)) + 1;
			double imax = CalculateGreatness.max_min(Big_Algae, 1);//max
			double imin = CalculateGreatness.max_min(Big_Algae, 0);//min
			double big_algae_to_1_arr[N];
			int index_max;
			int index_min;
			for (int i = 0; i < N; i++)
			{
				big_algae_to_1_arr[i] = Big_Algae[i];//reshape,array copy
				if (imax == Big_Algae[i])
					index_max = i;
				if (imin == Big_Algae[i])
					index_min = i;
			}
			if (m >= 40)
				m = m - 1;
			Algae[index_min][m] = Algae[index_max][m];
			starve = (int)(CalculateGreatness.max_min(Starve, 1));
			random = (rand() % 100000) / 100000.0;
			if (random < Ap)
				for (int m=0;m<D;m++)
					Algae[starve][m] = Algae[starve][m] + (Algae[index_max][m] - Algae[starve][m]) * ((rand() % 100000) / 100000.0);
			printf("Run = %d error = %1.8e\n", counter, Obj_Best_Algae);
		}
		return Obj_Best_Algae;
	}
};

float mean(double x[Nr]){
	float m_value=0;
	for (int i = 0; i < Nr; i++)
	{
		m_value = m_value + x[i];
	}
	return m_value/Nr;
	}

float std_value(double x[Nr]) {
	float mean_value = mean(x);

	float differ, varsum=0, variance, std;
	for (int i = 0; i < Nr; i++)
	{
		differ = x[i] - mean_value;
		varsum = varsum + pow(differ, 2);
	}

	variance = varsum / (float)Nr;
	std = sqrt(variance);
	return std;
}

int main()
{
	CalculateGreatness CalculateGreatness;
	srand(time(0));
	double F_RUNS[Nr];
	double Total_Time[Nr];
	float tic,toc;
	int counter = 0;
	float sum = 0, mean_value;
	AAA AAA;
	for (int i = 0; i < Nr; i++)
	{
		F_RUNS[i] = 0;
		Total_Time[i]=0;
	}

	for (int r = 0; r < Nr; r++)
	{
		tic = clock();
		counter = r + 1;
		F_RUNS[r]= AAA.AAA_();
		toc = clock();
		Total_Time[r] = toc - tic;
		printf("Run = %d error = %1.8e\n", counter, F_RUNS[r]);
	}
	printf("\n*************************\n");
	for (int i = 0; i < Nr; i++)
	{
		printf("%f ", F_RUNS[i]);
	}
	printf("\n*************************\n");
	printf("AAA\n");
	printf("AvgFitness = %1.10e BestFitness = %1.10e WorstFitness = %1.10e Std = %1.10e Median = %1.10e\n", mean(F_RUNS), CalculateGreatness.max_min(F_RUNS,0), CalculateGreatness.max_min(F_RUNS,1), std_value(F_RUNS), 0,9999999999);
	printf("Avg. time = %1.5e(%1.5e)\n",mean(Total_Time), std_value(Total_Time));
	printf("\n*************************\n");

	

}