#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h> 
#include <curand.h>
#include <curand_kernel.h>

int MaxFEVs = 1000;   // Maximum Fitness Calculation Number
const int N = 100;           // Number of Algal Colony in Population
const int D = 80;           // Number of Dimension in Problem
int LB = -100;        // Minimum Values for each dimension(ones(1, Parameters.D) *(-100);)
int UB = 100;        // Maximum Values for each dimension(ones(1, Parameters.D) * 100;)
int K = 2;           // Shear Force
double le = 0.3;         // Energy Loss
double Ap = 0.5;        // Adaptation
const int Nr = 5;  //No. of runs
#define PI 3.141592653589793

__global__ void setup_kernel(curandState * state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__global__ void generate(curandState* globalState,float *x)
{
	int ind = threadIdx.x;
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	*x = RANDOM;
	globalState[ind] = localState;
}

float rand_number() {
	curandState *devStates;
	float *x;
	float random;
	cudaMalloc(&x, sizeof(float));
	cudaMalloc(&devStates, sizeof(curandState));
	cudaMemcpy(x, &random, sizeof(float), cudaMemcpyHostToDevice);
	// setup seeds
	setup_kernel << < 1, 1 >> > (devStates, time(NULL));

	// generate random numbers
	generate << < 1, 1 >> > (devStates, x);
	cudaMemcpy(&random, x, sizeof(float), cudaMemcpyDeviceToHost);
	return random;
}

__global__ void Multiplication(double (*x)[D],double (*mul)[D]) {
	int j = threadIdx.x;
	int i = threadIdx.y;

	for (int k = 0; k < N; k++)
	{
		mul[i+k][j] = x[i+k][j] * x[i+k][j];
	}
}
__global__ void Multiplication1(double(*a)[D], double(*mul)[D]) {
	int i=threadIdx.x;
	if(i<D)
		mul[0][i] = a[0][i] * a[0][i];
}
__global__ void sum(double x[][D], double objValue[N]) {
	int total = 0;
	int j = threadIdx.x;
	int i = threadIdx.y;

	for (int k = 0; k < D; k++)
	{
		total += x[j][k];
		objValue[j] = total;
	}
}
/*_global__ void sum1(double x[][D], double objValue[1]) {
	int total = 0;
	int j;
	for (j = 0; j < D;j++) {
		total += x[0][j];
	}
	objValue[0] = total;

}*/
class Sphere {
public:
	
	double * ObjVal(double colony[][D]) {

		double mul[N][D];
		double (*x)[D];//colony
		double(*y)[D];//mul
		cudaMalloc(&x, N * D * sizeof(double));
		cudaMalloc(&y, N * D * sizeof(double));
		cudaMemcpy(x, colony, N * D * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(y, mul, N * D * sizeof(double), cudaMemcpyHostToDevice);
		Multiplication<<<1,D>>>(x,y);
		cudaMemcpy(mul, y, N * D * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(x);
		cudaFree(y);

		double objValue[N];
		double *z;
		cudaMalloc(&z, N * sizeof(double));
		cudaMalloc(&y, N * D * sizeof(double));
		cudaMemcpy(z, objValue, N * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(y, mul, N * D * sizeof(double), cudaMemcpyHostToDevice);
		sum << <1, N >> > (y, z);
		cudaMemcpy(objValue, z, N * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(y);
		cudaFree(z);

		return objValue;

	}
	double * sum1(double x[][D]) {//paralelleştirme gerekli değil//daha sonra ufak bir paralelleştirme yapılacak
		int total = 0;
		double objValue[1];
		for (int j = 0; j < D; j++)
			{
				total += x[0][j];
			}
		objValue[0] = total;
		
		return objValue;
	}
	double * ObjVal1(double colony[][D]) {

		double mul[1][D];
		double(*x)[D];//colony
		double(*y)[D];//mul
		cudaMalloc(&x, D * sizeof(double));
		cudaMalloc(&y, D * sizeof(double));
		cudaMemcpy(x, colony, D * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(y, mul, D * sizeof(double), cudaMemcpyHostToDevice);
		Multiplication1 << <1, D >> > (x, y);
		cudaMemcpy(mul, y, D * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(x);
		cudaFree(y);
		
		double objValue[1];
		/*double *z;
		cudaMalloc(&z, sizeof(double));
		cudaMalloc(&y, D * sizeof(double));
		cudaMemcpy(z, objValue, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(y, mul, D * sizeof(double), cudaMemcpyHostToDevice);
		sum1 << <1, D >> > (y, z);
		cudaMemcpy(objValue, z, sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(z);
		cudaFree(y);*/
		double* a = sum1(mul);
		objValue[0] = *(a);
		return objValue;

	}
};

__device__ double max_min2(double Obj_Algae[N], int x) {
	double minElement = Obj_Algae[0];
	double maxElement = Obj_Algae[0];

	for (int i = 0; i < N; i++) {
		if (Obj_Algae[i] < minElement)
			minElement = Obj_Algae[i];
		if (Obj_Algae[i] > maxElement)
			maxElement = Obj_Algae[i];
	}
	if (x == 1)
		return maxElement;
	else
		return minElement;
}
__device__ double ptp2(double Obj_Algae[N]) {
	return max_min2(Obj_Algae, 1) - max_min2(Obj_Algae, 0);
}
__global__ void forC1(double Obj_Algae[N], double ObjX[N]) {
	int i = threadIdx.x;
	ObjX[i] = (Obj_Algae[i] - max_min2(Obj_Algae, 0)) / ptp2(Obj_Algae);
}
__global__ void forC2(double ObjX[N], double Big_Algae[N]) {
	/*double fKs[N];
	double M[N];
	double dX[N];*/
	int i = threadIdx.x;
	double fKs = abs(Big_Algae[i] / 2.0);
	double M = (ObjX[i] / (fKs + ObjX[i]));

	double dX = M * Big_Algae[i];

	Big_Algae[i] = Big_Algae[i] + dX;
}
class CalculateGreatness {
public:
	void CalculateGreatness_(double Big_Algae[N], double Obj_Algae[N]) {
		double ObjX[N];
		double min = ptp(Obj_Algae);
		double * m;
		double * n;
		cudaMalloc(&m, N * sizeof(double));
		cudaMalloc(&n, N * sizeof(double));
		cudaMemcpy(m, Obj_Algae, N * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(n, ObjX, N * sizeof(double), cudaMemcpyHostToDevice);
		forC1 << <1, N >> > (m,n);
		cudaMemcpy(ObjX, n, N * sizeof(double), cudaMemcpyDeviceToHost);
		/*for (int i = 0; i < N; i++)//forC1
		{
			ObjX[i] = (Obj_Algae[i] - max_min(Obj_Algae, 0)) / ptp(Obj_Algae);
		}*/
		//s2 = N
		/*for (int i = 0; i < N; i++)//forC2
		{
			double fKs = abs(Big_Algae[i] / 2.0);

			double M = (ObjX[i] / (fKs + ObjX[i]));

			double dX = M * Big_Algae[i];

			Big_Algae[i] = Big_Algae[i] + dX;
		}*/
		cudaMemcpy(m, ObjX, N * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(n, Big_Algae, N * sizeof(double), cudaMemcpyHostToDevice);
		forC2 << <1, N >> > (m, n);
		cudaMemcpy(Big_Algae, n, N * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(m);
		cudaFree(n);
	}
	double max_min(double Obj_Algae[N], int x) {
		double minElement = Obj_Algae[0];
		double maxElement = Obj_Algae[0];

		for (int i = 0; i < N; i++) {
			if (Obj_Algae[i] < minElement)
				minElement = Obj_Algae[i];
			if (Obj_Algae[i] > maxElement)
				maxElement = Obj_Algae[i];
		}
		if (x == 1)
			return maxElement;
		else
			return minElement;
	}
	double ptp(double Obj_Algae[N]) {
		return max_min(Obj_Algae, 1) - max_min(Obj_Algae, 0);
	}
};
__global__ void forG1(double BAlgae_Great_Surface[N], double sorting[N]) {
	int i = threadIdx.x;
	BAlgae_Great_Surface[i] = 0;
	sorting[i] = i;
}
__global__ void forG2(double Big_Algae[N],double BAlgae_Great_Surface[N], double sorting[N]) {
	int i = threadIdx.x;
	int i_sort, j_sort, l_sort;
	for (int j = i + 1; j < N; j++)
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
	BAlgae_Great_Surface[l_sort] = i*i;
	
}
__global__ void clone(double x[N], double clone[N]) {
	int i = threadIdx.x;
	clone[i] = x[i];
}
__global__ void forGF3(double BAlgae_Surface[N], double copySurface[N]) {
	int i = threadIdx.x;
	BAlgae_Surface[i] = (copySurface[i] - max_min2(copySurface, 0)) / ptp2(copySurface);
}
class GreatnessOrder {
public:
	//s2=N
	CalculateGreatness CalculateGreatness;
	void GreatnessOrder_(double BAlgae_Great_Surface[N],double Big_Algae[N]) {
		double sorting[N];
		/*for (int i = 0; i < N; i++)//forG1
		{
			BAlgae_Great_Surface[i] = 0;
			sorting[i] = i;
		}*/
		double * m;
		double * n;
		cudaMalloc(&m, N * sizeof(double));
		cudaMalloc(&n, N * sizeof(double));
		cudaMemcpy(m, BAlgae_Great_Surface, N * sizeof(double), cudaMemcpyHostToDevice);//m=BAlgae_Great_Surface
		cudaMemcpy(n, sorting, N * sizeof(double), cudaMemcpyHostToDevice);//n=sorting
		forG1 << <1, N >> > (m, n);
		int i, k_sort;
		/*for (i = 0; i < N - 1; i++)//forG2
		{
			for (int j = i + 1; j < N; j++)
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
		}*/
		double * o;
		cudaMalloc(&o, N * sizeof(double));
		cudaMemcpy(o, Big_Algae, N * sizeof(double), cudaMemcpyHostToDevice);//o=Big_Algae
		forG2 << <1, N >> > (o,m, n);
		cudaMemcpy(sorting, n, N * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(BAlgae_Great_Surface, m, N * sizeof(double), cudaMemcpyDeviceToHost);
		k_sort = (int)(sorting[N - 2]);
		BAlgae_Great_Surface[k_sort] = (N - 1) * (N - 1);
		double copySurface[N];
		/*for (int i = 0; i < N; i++)//clone
		{
			copySurface[i] = BAlgae_Great_Surface[i];
		}*/
		cudaMemcpy(n, copySurface, N * sizeof(double), cudaMemcpyHostToDevice);//n=copySurface
		clone << <1, N >> > (m, n);
		cudaMemcpy(copySurface, n, N * sizeof(double), cudaMemcpyDeviceToHost);
		/*for (int i = 0; i < N; i++)//forGF3
		{

			BAlgae_Great_Surface[i] = (copySurface[i] - CalculateGreatness.max_min(copySurface, 0)) / CalculateGreatness.ptp(copySurface);
		}*/
		forGF3 << <1, N >> > (m, n);
		cudaMemcpy(BAlgae_Great_Surface, m, N * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(m);
		cudaFree(n);
		cudaFree(o);
	}
};
__global__ void forF1(double Big_Algae[N],double BAlgae_Fr_Surface[N]) {
	int i = threadIdx.x;
	double r;
	BAlgae_Fr_Surface[i] = 0;
	r = pow(((Big_Algae[i] * 3) / (4 * PI)), 0.333333); // Calculate the Radius
	double power = r * r;
	BAlgae_Fr_Surface[i] = 2 * PI * power;// Calculate the Friction Surface
}
class FrictionSurface {
public:
	//s2=N
	CalculateGreatness CalculateGreatness;
	void FrictionSurface_(double BAlgae_Fr_Surface[N],double Big_Algae[N]) {
		double r;
		double * m;
		cudaMalloc(&m, N * sizeof(double));
		cudaMemcpy(m, BAlgae_Fr_Surface, N * sizeof(double), cudaMemcpyHostToDevice);//m=BAlgae_Fr_Surface
		double * o;
		cudaMalloc(&o, N * sizeof(double));
		cudaMemcpy(o, Big_Algae, N * sizeof(double), cudaMemcpyHostToDevice);//o=Big_Algae
		/*for (int i = 0; i < N; i++)//forF1
		{
			BAlgae_Fr_Surface[i] = 0;
			r = pow(((Big_Algae[i] * 3) / (4 * PI)), 0.333333); // Calculate the Radius
			double power = pow(r, 2);
			BAlgae_Fr_Surface[i] = 2 * PI * power;// Calculate the Friction Surface
		}*/
		forF1 << <1, N >> > (o, m);
		cudaMemcpy(BAlgae_Fr_Surface, m, N * sizeof(double), cudaMemcpyDeviceToHost);
		double copySurface[N];
		double *n;
		cudaMalloc(&n, N * sizeof(double));
		cudaMemcpy(n, copySurface, N * sizeof(double), cudaMemcpyHostToDevice);//n=copySurface
		/*for (int i = 0; i < N; i++)//clone
		{
			copySurface[i] = BAlgae_Fr_Surface[i];
		}*/
		clone << <1, N >> > (m,n);
		cudaMemcpy(copySurface, n, N * sizeof(double), cudaMemcpyDeviceToHost);
		/*for (int i = 0; i < N; i++)//forGF3
		{
			BAlgae_Fr_Surface[i] = (copySurface[i] - CalculateGreatness.max_min(copySurface, 0)) / CalculateGreatness.ptp(copySurface);
		}*/
		forGF3 << <1, N >> > (m, n);
		cudaMemcpy(BAlgae_Fr_Surface, m, N * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(m);
		cudaFree(n);
		//cudaFree(o);//aktif olduğunda garip bir yavaşlama görüldü
	}
};

/*__global__ void randomPermutationN(int random[N],int x) {
	int i = threadIdx.x;
	int index;
	random[i] = i;
	double temp;
	index = rand() % x;
	temp = random[i];
	random[i] = random[index];
	random[index] = temp;
}*/
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
int * randomPermutationD(int x) {//||||||||||||||||||||||||||||
	int index;
	int random[D];

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

class Tournement_selection {
public:
	int Tournement_selection_(double Obj_Algae[N]) {
		//s2=N
		double choice;
		int neighbor[N];
		int * x = randomPermutationN(N);
		for (int i = 0; i < N; i++)
			neighbor[i] = *(x + i);
		/*int * x;
		cudaMalloc(&x, N * sizeof(int));
		cudaMemcpy(x, neighbor, N * sizeof(int), cudaMemcpyHostToDevice);
		randomPermutationN << <1, N >> > (x, N);
		cudaMemcpy(neighbor, x, N * sizeof(double), cudaMemcpyDeviceToHost);*/
		if (Obj_Algae[neighbor[0]] < Obj_Algae[neighbor[1]])
			choice = neighbor[0];
		else
			choice = neighbor[1];
		return choice;
	}
};
/*__global__ void forA1(double Obj_Algae[], int * indices, double * Obj_Best_Algae) {
	int i = threadIdx.x;
	double min_Obj_Alg = max_min2(Obj_Algae, 0);
	if (min_Obj_Alg == Obj_Algae[i]) {
		printf("efsd %d", i);
		indices = &i;
		Obj_Best_Algae = &min_Obj_Alg;
	}
}*/
__global__ void forA2(double Best_Algae[], double Algae[D]) {
	int i = threadIdx.x;
	Best_Algae[i] = Algae[i];//Best_Algae = Algae[indices, :]
}
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
		double Obj_New_Algae[1];
		double random;
		int q = 0;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < D; j++)
			{
				random = (rand() % 1000) / 1000.0;
				//double r = (rand() % 1000000) / 10000.0;
				Algae[i][j] = LB + (UB - LB) * random;
			}
			Starve[i] = 0;
			Big_Algae[i] = 1;
		}

		Sphere sphere;
		CalculateGreatness CalculateGreatness;
		GreatnessOrder GreatnessOrder;
		FrictionSurface FrictionSurface;
		Tournement_selection Tournement_selection;
		int parameters[D];
		double * x = sphere.ObjVal(Algae);
		for (int i = 0; i < N; i++)
			Obj_Algae[i] = *(x + i);
		double min_Obj_Alg = CalculateGreatness.max_min(Obj_Algae, 0);//np.min
		double Obj_Best_Algae;
		int indices;
		for (int i = 0; i < N; i++)//forA1
		{
			if (min_Obj_Alg == Obj_Algae[i]) {
				indices = i;
				Obj_Best_Algae = min_Obj_Alg;
			}
		}
		/*int * ind;
		double * o;
		double * oba;
		cudaMalloc(&ind, sizeof(int));
		cudaMalloc(&o, N * sizeof(double));
		cudaMalloc(&oba, sizeof(double));
		cudaMemcpy(&o, Obj_Algae, N * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ind, &indices, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(oba, &Obj_Best_Algae, sizeof(double), cudaMemcpyHostToDevice);
		forA1 << <1, N >> > (o, ind, oba);
		cudaMemcpy(&Obj_Best_Algae, oba, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(&indices, ind, sizeof(int), cudaMemcpyDeviceToHost);*/
		/*for (int i = 0; i < D; i++)//forA2
		{
			Best_Algae[i] = Algae[indices][i];//Best_Algae = Algae[indices, :]
		}*/
		double * b;
		double * a;
		cudaMalloc(&b, D * sizeof(double));
		cudaMalloc(&a, D * sizeof(double));
		cudaMemcpy(b, Best_Algae, D * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(a, Algae[indices], D * sizeof(double), cudaMemcpyHostToDevice);
		forA2 << <1, D >> > (b,a);
		cudaMemcpy(Best_Algae, b, D * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(a);
		cudaFree(b);
		CalculateGreatness.CalculateGreatness_(Big_Algae, Obj_Algae);
		int counter = 0;
		int c = N, i, starve;

		while (c < MaxFEVs) {

			GreatnessOrder.GreatnessOrder_(Cloro_ALG,Big_Algae); //Calculate energy values
			FrictionSurface.FrictionSurface_(Big_Algae_Surface,Big_Algae); //Sorting by descending size and normalize between[0, 1]

			for (i = 0; i < N; i++)
			{
				starve = 0;
				while (Cloro_ALG[D - 1] >= 0 && c < MaxFEVs) {
					Neighbor = Tournement_selection.Tournement_selection_(Obj_Algae);////********/////
					while (Neighbor == i) {
						Neighbor = Tournement_selection.Tournement_selection_(Obj_Algae);
					}
					int * t = randomPermutationD(D);
					for (int r = 0; r < D; r++)
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
					double random = (rand() % 1000000) / 1000000.0;
					double rand_value = random - 0.5;
					random = (rand() % 1000000) / 1000000.0;
					double cosine_value = cos(random * 360);
					random = (rand() % 1000000) / 1000000.0;
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

					double * x = sphere.ObjVal1(New_Algae);
					
					Obj_New_Algae[0] = *(x);
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
				Starve[i - 1] = 1;
			}
			//[val, ind] = np.min(Obj_Algae)
			double min_Obj_Alg1 = CalculateGreatness.max_min(Obj_Algae, 0);
			double valki=0;
			int ind=0;

			for (int i = 0; i < N; i++)
			{
				if (min_Obj_Alg1 == Obj_Algae[i]) {
					ind = i;
					valki = min_Obj_Alg1;
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
			double random = (rand() % 1000000) / 1000000.0;
			int m = ((int)(random * D)) + 1;
			double imax = CalculateGreatness.max_min(Big_Algae, 1);//max
			double imin = CalculateGreatness.max_min(Big_Algae, 0);//min
			double big_algae_to_1_arr[N];
			int index_max=0;
			int index_min=0;
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
			random = (rand() % 1000000) / 1000000.0;
			if (random < Ap)
				for (int m = 0; m < D; m++)
					Algae[starve][m] = Algae[starve][m] + (Algae[index_max][m] - Algae[starve][m]) * ((rand() % 1000000) / 100000.0);
			printf("Run = %d error = %1.8e\n", counter, Obj_Best_Algae);
		}
		return Obj_Best_Algae;
	}
};

float mean(double x[Nr]) {
	float m_value = 0;
	for (int i = 0; i < Nr; i++)
	{
		m_value = m_value + x[i];
	}
	return m_value / Nr;
}

float std_value(double x[Nr]) {
	float mean_value = mean(x);

	float differ, varsum = 0, variance, std;
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
	float tic, toc;
	int counter = 0;
	float sum = 0, mean_value;
	AAA AAA;
	for (int i = 0; i < Nr; i++)
	{
		F_RUNS[i] = 0;
		Total_Time[i] = 0;
	}

	for (int r = 0; r < Nr; r++)
	{
		tic = clock();
		counter = r + 1;
		F_RUNS[r] = AAA.AAA_();
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
	double bestFitness = CalculateGreatness.max_min(F_RUNS, 0);
	double worstFitness = CalculateGreatness.max_min(F_RUNS, 1);
	double std = std_value(F_RUNS);
	printf("AvgFitness = %1.10e BestFitness = %1.10e WorstFitness = %1.10e Std = %1.10e Median = %1.10e\n", mean(F_RUNS), bestFitness, worstFitness, std, 0, 9999999999);
	printf("Avg. time = %1.5e(%1.5e)\n", mean(Total_Time), std_value(Total_Time));
	printf("\n*************************\n");


}

  
