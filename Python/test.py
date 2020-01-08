import time
import numpy as np
from AAA import AAA
from TestFunctions import FnList, FnParams

# Parameters to be tested
MaxFEVs_list = [1000, 2000, 3000, 4000, 5000]
N_list = [20, 40, 60, 80, 100]
D_List = [10, 30, 50, 70, 90]

# As defined by original algo
K = 2 #2            # Shear Force
le = 0.3         # Energy Loss
Ap = 0.5         # Adaptation
Nr = 100          # No. of runs

with open("AAA_results.csv", "w") as f:
    f.write("Function,MaxFEVs,N,D,Run,Result,Time\n")

for fn_i, fn in enumerate(FnList):
    print("\n*************************")
    print("*** Running function {}/{}: {} {}".format(fn_i + 1, len(FnList), fn.__name__, FnParams[fn.__name__]))
    print("*************************\n")

    LB = FnParams[fn.__name__][1]
    UB =  FnParams[fn.__name__][2]

    for MaxFEVs in MaxFEVs_list:
        for N in N_list:
            for D in D_List:
                print("** Params: MaxFEVs = {} N = {} D = {}".format(MaxFEVs, N, D))
                
                F_RUNS = np.zeros(Nr)
                Total_Time = np.zeros(Nr)
                
                for r in range(0, Nr):                    
                    tic = time.time()  # Reset the timer
                    F_RUNS[r] = AAA(MaxFEVs * D, N, D, LB, UB, K, le, Ap, fn)
                    toc = time.time() - tic
                    
                    Total_Time[r] = toc
                    #print("Run = %d error = %1.8e"% (r + 1 , F_RUNS[:,r]))
                
                    #if(r%10 == 9):
                    print('.'*r, end='\r')

                with open("AAA_results.csv", "a") as f:
                    for i, res in enumerate(F_RUNS):
                        f.write("{},{},{},{},{},{},{}\n".format(fn.__name__, MaxFEVs, N, D, i + 1, res, Total_Time[i]))

                print("\n* Fitness: Avg = %1.10e Best = %1.10e Worst = %1.10e Std = %1.10e Median = %1.10e" %(np.mean(F_RUNS), np.min(F_RUNS), np.max(F_RUNS), np.std(F_RUNS), np.median(F_RUNS)))
                print("* Time: Avg = %1.5e(%1.5e)\n" %(np.mean(Total_Time), np.std(Total_Time)))
