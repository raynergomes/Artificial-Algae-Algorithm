import time
import numpy as np
from AAA import AAA
from TestFunctions import ObjVal

MaxFEVs = 1000   # Maximum Fitness Calculation Number
N = 50 #50           # Number of Algal Colony in Population
D = 40 #40           # Number of Dimension in Problem
LB = -100 #-100        # Minimum Values for each dimension (ones(1 ,Parameters.D ) *(-100);)
UB =  100 #100        # Maximum Values for each dimension (ones(1 ,Parameters.D ) *100;)
K = 2 #2            # Shear Force
le = 0.3         # Energy Loss
Ap = 0.5         # Adaptation

Nr = 10          # No. of runs

F_RUNS = np.zeros((1, Nr))
Total_Time = np.zeros((1, Nr))

for r in range(0, Nr):
    tic = time.time()  # Reset the timer
    counter = r + 1
    F_RUNS[:,r] = AAA(MaxFEVs * D, N, D, LB, UB, K, le, Ap, ObjVal)

    toc = time.time() - tic
    Total_Time[:,r] = toc
    print("Run = %d error = %1.8e\n"% (counter , F_RUNS[:,r]))

print("\n*************************\n")
print("AAA\n")
print("AvgFitness = %1.10e BestFitness = %1.10e WorstFitness = %1.10e Std = %1.10e Median = %1.10e\n" %(np.mean(F_RUNS), np.min(F_RUNS), np.max(F_RUNS), np.std(F_RUNS), np.median(F_RUNS)));
print("Avg. time = %1.5e(%1.5e)\n" %(np.mean(Total_Time), np.std(Total_Time)))
print("\n*************************\n")
