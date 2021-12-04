# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:45:30 2021

@author: Rahouti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import datasets
from sklearn.mixture import GaussianMixture

# chargement du jeu données iris
iris = datasets.load_iris()

# selecter les 2 premières colonnes
X = iris.data[:, :2]

# transformer les données en type dataframe
d = pd.DataFrame(X)

# plot des données
plt.scatter(d[0], d[1])

plt.show()

gmm = GaussianMixture(n_components = 4)

# Fit the GMM model for the dataset
# which expresses the dataset as a
# mixture of 3 Gaussian Distribution
gmm.fit(d)

# Assign a label to each sample
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
d3 = d[d['labels']== 3]

# plot three clusters in same plot
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')
plt.scatter(d3[0], d3[1],c='black')
plt.show()