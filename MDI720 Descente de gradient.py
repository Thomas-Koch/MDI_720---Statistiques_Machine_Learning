# -*- coding: utf-8 -*-
"""
Descente de gradient en 1D avec constante 
avec les données des slides de MDI 720 Cours 2
"""

import pandas as pd
import numpy as np
from numpy.linalg import norm
import statsmodels.api as sm

# Load data
url = 'https://forge.scilab.org/index.php/p/rdataset/source/file/master/csv/datasets/cars.csv'
dat = pd.read_csv(url)
y = dat['dist']
X = dat['speed']
X=X.values # on passe du dataframe vers un array
X = sm.add_constant(X) # ajout d'une colonne de 1 dans X
y=y.values

"""
Descente de gradient
L'arrêt est basé sur la norme du gradient
"""

epsilon = 1e-7 # borne limite déclenchant l'arrêt
alpha = 1e-4 # pas : coefficient du multiplication du gradient
theta = np.array([-17.0,3.9]) # valeur initiale
nbcycle = 0 # compteur

# on compare epsilon à la norme du gradient
while norm(np.dot(np.dot(X.T,X),theta)-np.dot(X.T,y)) > epsilon:
    theta = theta - alpha * (np.dot(np.dot(X.T,X),theta)-np.dot(X.T,y))
    nbcycle = nbcycle +  1
print("nb cycles", nbcycle)
print("descente de gradient", theta)

       
