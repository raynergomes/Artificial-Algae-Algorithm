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
    return 0

def F05(Colony):
    """ Ackley """
    return 0

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
    prod = np.array(range(S.shape[-1]))
    S = np.multiply(S, np.tile(prod, (S.shape[0], 1)))   

    if dim == 2:
        S = np.sum(S.transpose(), 0)
    else:
        S = np.sum(S)
    return S

def ObjVal(Colony):
    return F03(Colony)
    
FnList = [F01, F02, F03, F06, F07]
FnParams = {
    "F01": ("sphere", -100, 100),
    "F02": ("Rosenbrock", -10, 10),
    "F03": ("Rastrigin", -5.12, 5.12),
    "F04": ("Griewank", -600, 600),
    "F05": ("Ackley", -10, 10),
    "F06": ("Schwefel", -500, 500),
    "F07": ("Sum Squares", -100, 100)
}