#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################
# Created on Thu Jun 13 12:32:07 2019
#
# @author: jaman1@llnl.gov
# Resource used: RandomForestSnippet.py (loveland4@llnl.gov)
###############################################################

import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#import os
import sys
import pandas as pd 
import numpy as np
#import scipy

def LG(data, leaveout_lot):  
        """
        Run a Linear regression model on the data

        Arguments:
        data: The dataframe you will be using for your regression (pandas)
        leaveout_lot: The lot you will be leaving out (string)
       
        """
        sys.stdout.flush()
        training = data.values[data.index!=leaveout_lot]
        testing = data.values[data.index==leaveout_lot]
        #verbose=0
        poly = PolynomialFeatures(degree=1)
        model = LinearRegression()
        
        input_train_data = training[:,0:2]
        output_train_data = training[:,2]     
        input_train_data = poly.fit_transform(input_train_data)
        
        input_test_data = testing[:,0:2]
        input_test_data = poly.fit_transform(input_test_data)
        #output_test_data=testing[:,2]
 
        model.fit(input_train_data, output_train_data)
        predicted_stress = model.predict(input_test_data)
        error=abs(testing[:,2]-predicted_stress)
        pError=100*error/testing[:,2]
        # Round to 2 decimal place
        temp = {'Slope':{leaveout_lot:testing[:,0].round(2)}, 'Strain':{leaveout_lot:testing[:,1].round(2)}, 'Stress':{leaveout_lot:testing[:,2].round(2)}, 'Predicted Stress':{leaveout_lot:predicted_stress.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info

#SEM_output_data.pkl should be in working directory
pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
print("\nPrinting raw dictionary: \n")
print(sem_data)

dataset=pd.DataFrame.from_dict(sem_data)
print("\nPrinting dataframe extracted from dictionary: \n")
print(dataset)
print("\nPrinting generic statistics: \n")
print(dataset.describe())
print("\n\n")

result=LG(dataset, dataset.index[0])
for i in dataset.index:
    temp=LG(dataset, i)
    result=result.append(temp)
result = result[1:dataset.index.size+1]
print(result)

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in result.index:
    MAPE+=result.loc[i,'%Error']
    RMSE+=(result.loc[i,'Absolute error'])**2
    MGT+=result.loc[i,'Stress']
MAPE/=result.index.size
RMSE/=result.index.size
RMSE=RMSE**(0.5)
MGT/=result.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

#Plot/Compare
sort=result.sort_values(by="%Error")
plt.plot(sort.loc[:,'%Error'],'g-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error')
plt.title('Prediction error: LG')
plt.legend('L')
plt.grid()
plt.show()
