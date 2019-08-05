# -*- coding: utf-8 -*-
#######################
# Golam Gause Jaman   #
#######################

import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

#SEM_output_data.pkl should be in working directory
pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
print("Printing raw dictionary: \n")
print(sem_data)

dataset=pd.DataFrame.from_dict(sem_data)
print("\nPrinting dataframe extracted from dictionary: \n")
print(dataset)
print("\nPrinting generic statistics: \n")
print(dataset.describe())
print("\nScatter matrix: \n")
scatter_matrix(dataset)
plt.show()
print("\nCorrelation coefficients: \n")
print(dataset.corr())
