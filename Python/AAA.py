import numpy as np

from CalculateGreatness import CalculateGreatness
from FrictionSurface import FrictionSurface
from GreatnessOrder import GreatnessOrder
from Sphere import ObjVal
from tournement_selection import tournement_selection


def AAA(MaxFEVs, N, D, LB, UB, K, le, Ap):
  Algae = np.zeros((N, D))
  Algae = LB + (UB - LB) * np.random.rand(N, D)

  # N é o numero de algas - tamanho da populacao
  Starve = np.zeros((1, N))
  # Inicia com um, pois 1 significa o menor valor dentro deste array.
  # Quando calcular o CalculateGreatness, todos os valores serao maior ou igual a 1, o 1 será o menor!
  Big_Algae = np.ones((1, N))
  Obj_Algae = []
  Best_Algae = np.zeros((1, N))

  for i in range(1, N):
    Obj_Algae = ObjVal(Algae)

  value = np.min(Obj_Algae)
  indices = np.where(Obj_Algae == value)  # ele retorna um array por isto do [0]

  Best_Algae = Algae[indices, :]
  Obj_Best_Algae = value

  Big_Algae = CalculateGreatness(Big_Algae, Obj_Algae)
  counter = 0

  c = N  # N is number of algae
  while c < MaxFEVs:  # Maximum Fitness Calculation Number
    #  Evaluate energy and friction of n algae
    Cloro_ALG = GreatnessOrder(Big_Algae);  # Calculate energy values
    Big_Algae_Surface = FrictionSurface(Big_Algae);  # Sorting by descending size and normalize between[0, 1]

    for i in range(0, c):  # c = N = tamanho da populacao
      starve = 0  # starvation is ture
      while Cloro_ALG[:, D - 1] >= 0 and c < MaxFEVs:
        # Choose j among all solutions via tournament selection method
        while (Neighbor := tournement_selection(Obj_Algae)) == i:
          Neighbor = tournement_selection(Obj_Algae)

        # Choose randomly three dimensions to helical movement
        parameters = np.random.permutation(D)
        parameter0 = np.int(parameters[0])
        parameter1 = np.int(parameters[1])
        parameter2 = np.int(parameters[2])

        New_Algae = Algae[i, :]
        Subtr_Eq0 = np.float(Algae[Neighbor, parameter0] - New_Algae[parameter0])
        Subtr_Eq1 = np.float(Algae[Neighbor, parameter0] - New_Algae[parameter0])
        Subtr_Eq2 = np.float(Algae[Neighbor, parameter0] - New_Algae[parameter0])

        # K = shear force - tau na fórmula
        K_Big_Algae = K - np.float(Big_Algae_Surface[:, i])

        rand_value = np.random.random() - 0.5
        cosine_value = np.cos(np.random.random() * 360)
        sine_value = np.sin(np.random.random() * 360)

        New_Algae[parameter0] = Subtr_Eq0 * K_Big_Algae * (rand_value * 2)
        New_Algae[parameter1] = Subtr_Eq1 * K_Big_Algae * cosine_value
        New_Algae[parameter2] = Subtr_Eq2 * K_Big_Algae * sine_value

        # Bounding
        for p in range(1, 3):
          if New_Algae[parameters[p]] > UB:
            New_Algae[parameters[p]] = UB
          if New_Algae[parameters[p]] < LB:
            New_Algae[parameters[p]] = LB

        Obj_New_Algae = ObjVal(New_Algae)

        c = c + 1
        counter = c
        # No pseudo-codigo o Cloro_ALG é o E(xi) = Energy Loss caused by movement
        Cloro_ALG[:, i] = Cloro_ALG[:, i] - (le / 2)

        if Obj_New_Algae <= Obj_Algae[i]:
          Algae[i, :] = New_Algae
          Obj_Algae[i] = Obj_New_Algae
          starve = 1
        else:
          Cloro_ALG[:, i] = Cloro_ALG[:, i] - (le / 2)

    if starve == 0:
      Starve[:, i] = Starve[:, i] + 1

    # [val, ind] = np.min(Obj_Algae)
    valki = np.min(Obj_Algae)
    ind = np.where(Obj_Algae == valki)

    # make the biggest equal the mim
    if valki < Obj_Best_Algae:
      Best_Algae = Algae[ind, :]
      Obj_Best_Algae = valki

    Big_Algae = CalculateGreatness(Big_Algae, Obj_Algae)

    # Choose one dimension to reproduction, r
    # Round an array of floats element-wise to nearest integer towards zero.
    m = np.int(np.fix(np.random.random() * D) + 1)
    if m >= D:
      m = m - 1;

    big_algae_to_1_arr = np.array(Big_Algae).reshape(N)
    # Algae[imin, m] = Algae[imax, m]

    # descobrir qual posicao está o max e min
    maxi_value = np.max(big_algae_to_1_arr)
    index_max = np.where(big_algae_to_1_arr == maxi_value)
    mini_value= np.min(big_algae_to_1_arr)
    index_min = np.where(big_algae_to_1_arr == mini_value )

    Algae[index_min, m] = Algae[index_max, m]

    starve = np.int(np.max(Starve))

    if np.random.random() < Ap:  # Adaptation, parameter
      for m in range(0, D):
        Algae[starve, m] = Algae[starve, m] + (Algae[index_max, m] - Algae[starve, m]) * np.random.random()

    print('Run = %d error = %1.8e\n' % (counter, Obj_Best_Algae))

  return Obj_Best_Algae
