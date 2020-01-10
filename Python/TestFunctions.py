"""
https://www.sfu.ca/~ssurjano/optimization.html
https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/

"""

import math
import numpy as np

def F01_(Colony):
    S = Colony * Colony
    sh = np.array(S).transpose()
    objValue = sum(sh)

    print("### SHAPES ###")
    print(Colony.shape)
    print(S.shape)
    print(sh.shape)
    print(objValue.shape)

    return objValue

def F01(Colony):
    """ Sphere """
    S = np.square(Colony)
    S = np.sum(S.transpose(), 0)
    return S

def F02(Colony):
    """ Rosenbrock """
    dim = len(Colony.shape)
    if dim == 2:
        Mat1 = Colony[:, 0:Colony.shape[1]-1]
        Mat2 = Colony[:, 1:Colony.shape[1]]
    else:
        Mat1 = Colony[0:Colony.shape[0]-1]
        Mat2 = Colony[1:Colony.shape[0]]
    
    S = 100*np.square(Mat2-np.square(Mat1))+np.square(1-Mat1)
    S = np.sum(S.transpose(), 0)
    return S

def F03(Colony):
    """ Rastrigin """
    dim = len(Colony.shape)
    if dim == 2:
        DD = Colony.shape[1]
    else:
        DD = Colony.shape[0]
    A = 10
    Omega = 2 * math.pi
    S = np.square(Colony) - A * np.cos(Omega * Colony)
    S = DD * A + np.sum(S.transpose(),0)
    return S

def F04(Colony):
    """ Griewank """
    dim = len(Colony.shape)
    S = np.divide(np.square(Colony),4000)
    S = np.sum(S.transpose(), 0)

    prod = np.sqrt(np.arange(1, Colony.shape[-1] + 1))
    if dim == 2:        
        prod = np.tile(prod, (Colony.shape[0], 1))
    P = np.divide(Colony, prod, out=np.zeros(Colony.shape, dtype=float), where=prod!=0)
    P = np.prod(P.transpose(), 0)

    S = S - P + 1
    return S

def F05(Colony):
    """ Ackley """
    a = 20
    b = 0.2
    c = 2 * math.pi
    d = Colony.shape[-1]
    dim = len(Colony.shape)
    S1 = np.square(Colony)
    S2 = np.cos(c * Colony)

    if dim == 2:
        S1 = np.sum(S1.transpose(), 0)
        S2 = np.sum(S2.transpose(), 0)
    else:
        S1 = np.sum(S1)
        S2 = np.sum(S2)
    
    S1 = -b * np.sqrt((1/d) * S1)
    S2 = (1/d) * S2

    S = -a * np.exp(S1) - np.exp(S2) +a - math.exp(1)

    return S

def F06(Colony):
    """ Schwefel """
    dim = len(Colony.shape)
    if dim == 2:
        DD = Colony.shape[1]
    else:
        DD = Colony.shape[0]
    
    A = DD * Colony.shape[-1]
    S = np.multiply(Colony, np.sin(np.sqrt(np.abs(Colony))))
    S = A - np.sum(S.transpose(),0)
    return S

def F07(Colony):
    """ Sum Squares """
    dim = len(Colony.shape)
    S = np.square(Colony)
    prod = np.arange(1, S.shape[-1] + 1)
    S = np.multiply(S, np.tile(prod, (S.shape[0], 1)))   

    if dim == 2:
        S = np.sum(S.transpose(), 0)
    else:
        S = np.sum(S)
    return S

def F08(Colony):
    """ Sum of Different Powers """
    dim = len(Colony.shape)
    pow_ = np.arange(1, Colony.shape[-1] + 1) + 1
    S = np.power(Colony, np.tile(pow_, (Colony.shape[0], 1)))
    if dim == 2:
        S = np.sum(S.transpose(), 0)
    else:
        S = np.sum(S)
    return S

def F09(Colony):
    """ Zakharov """
    dim = len(Colony.shape)
    S1 = np.sum(np.square(Colony).transpose(), 0)
    prod = 0.5 * np.arange(1, Colony.shape[-1] + 1)
    S2 = np.multiply(np.tile(prod, (Colony.shape[0], 1)), Colony)   
    
    if dim == 2:
        S2 = np.sum(S2.transpose(), 0)
    else:
        S2 = np.sum(S2)
    
    S = S1 + np.square(S2) + np.power(S2, 4)
    return S

def F10(Colony):
    """ Styblinski-Tang """
    S = np.power(Colony, 4) - 16 * np.square(Colony) + 5 * Colony
    S = 0.5 * np.sum(S.transpose(), 0)
    return S

def F11(Colony):
    """ Rotated Hyper-Ellipsoid Function """
    dim = len(Colony.shape)
    S = np.zeros(Colony.shape, dtype=float)
    if dim == 2:
        d = Colony.shape[1]
        for _ in range(d):
            S[:,_] = F01(Colony[:,:_ + 1]).transpose()
    else:
        d = Colony.shape[0]
        for _ in range(d):
            S[_] = F01(Colony[:_ + 1]).transpose()
                
    S = np.sum(S.transpose(), 0)
    return S

def F12(Colony):
    """ Trid """
    dim = len(Colony.shape)
    Mat0 = np.square(Colony-1)
    if dim == 2:
        Mat1 = Colony[:, 0:Colony.shape[1]-1]
        Mat2 = Colony[:, 1:Colony.shape[1]]
    else:
        Mat1 = Colony[0:Colony.shape[0]-1]
        Mat2 = Colony[1:Colony.shape[0]]
    
    S = np.sum(Mat0.transpose(), 0) - np.sum(np.multiply(Mat1,Mat2).transpose(), 0)
    return S

def F13(Colony):
    """ Quartic """
    dim = len(Colony.shape)
    S = np.power(Colony, 4)
    prod = np.arange(1, S.shape[-1] + 1)
    S = np.multiply(S, np.tile(prod, (S.shape[0], 1)))   

    if dim == 2:
        S = np.sum(S.transpose(), 0)
    else:
        S = np.sum(S)
    return S + np.random.random_sample()
    
FnList = [F01, F02, F03, F04, F05, F06, F07, F08, F09, F10, F11, F12, F13]
FnParams = {
    "F01": ("sphere", -5.12, 5.12), # -100, 100
    "F02": ("Rosenbrock", -5, 10), # -10, 10
    "F03": ("Rastrigin", -5.12, 5.12),
    "F04": ("Griewank", -600, 600),
    "F05": ("Ackley", -32.768, 32.768),
    "F06": ("Schwefel", -500, 500),
    "F07": ("Sum Squares", -5.12, 5.12),
    "F08": ("Sum of Different Powers", -1, 1),
    "F09": ("Zakharov", -5, 10),
    "F10": ("Styblinski-Tang", -5, 5),
    "F11": ("Rotated Hyper-Ellipsoid Function", -65.536, 65.536),
    "F12": ("Trid", -10, 10),
    "F13": ("Quartic", -5.12, 5.12)
}

def ObjVal(Colony):
    return F08(Colony)