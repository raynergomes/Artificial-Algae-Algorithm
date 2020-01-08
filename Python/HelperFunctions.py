import numpy as np

def CalculateGreatness(BigX,ObjX):

    ObjX = (ObjX - np.min(ObjX))/ np.ptp(ObjX)

    s2 = np.size(BigX)

    BigY = []

    for i in range(0,s2):

        fKs = np.abs(BigX[:,i]/2.0)

        M = (ObjX[i] / (fKs + ObjX[i]))

        dX = M * BigX[:,i]

        BigX[:,i] = BigX[:,i] + dX

    return BigX

def GreatnessOrder(Big_algae):
    s2 = np.size(Big_algae)

    sorting = np.ones((1, s2))

    BAlgae_Great_Surface = np.zeros((1, s2))

    for i in range(0,s2):
        sorting[:,i] = i

    for i in range(0,s2 - 1):

        for j in range(i + 1, s2):

            i_sort = np.int(sorting[:,i])
            j_sort = np.int(sorting[:,j])

            if Big_algae[:,i_sort] > Big_algae[:,j_sort]:

                temp = np.int(sorting[:,i])

                sorting[:,i] = sorting[:,j]

                sorting[:,j] = temp
        l_sort = np.int(sorting[:,i])
        BAlgae_Great_Surface[:,l_sort] = np.power(i, 2)

    k_sort = np.int(sorting[:,s2-2])
    BAlgae_Great_Surface[:,k_sort] = np.power((i + 1), 2)
    BAlgae_Great_Surface = (BAlgae_Great_Surface - np.min(BAlgae_Great_Surface))/ np.ptp(BAlgae_Great_Surface)

    return BAlgae_Great_Surface

def FrictionSurface(Big_Algae):

    s2 = np.size(Big_Algae)

    BAlgae_Fr_Surface = np.zeros((1, s2))

    for i in range(0,s2):
        r = np.power(((Big_Algae[:,i] * 3) / (4 * np.pi)), (1 / 3))  # Calculate the Radius
        BAlgae_Fr_Surface[:,i] = 2 * np.pi * np.power(r, 2)  # Calculate the Friction Surface

    BAlgae_Fr_Surface = (BAlgae_Fr_Surface - np.min(BAlgae_Fr_Surface)) / np.ptp(BAlgae_Fr_Surface)

    return BAlgae_Fr_Surface

def tournement_selection(source):

    s2 = np.size(source)
    #np.random.seed(10)
    neighbor = np.random.permutation(s2)

    if source[neighbor[0]] < source[neighbor[1]]:
        choice = neighbor[0]
    else:
        choice = neighbor[1]
    return choice