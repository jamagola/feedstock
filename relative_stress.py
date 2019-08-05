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
from matplotlib.pyplot import figure
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#import os
import sys
import pandas as pd 
import numpy as np
#import scipy

def GBR(data, leaveout_lot):  

        sys.stdout.flush()
        training = data.values[data.index!=leaveout_lot]
        testing = data.values[data.index==leaveout_lot]
        #verbose=0
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
        model = ensemble.GradientBoostingRegressor(**params)
        
        input_train_data = training[:,0:2]
        output_train_data = training[:,2]
        
        input_test_data=testing[:,0:2]
        #output_test_data=testing[:,2]
        model.fit(input_train_data, output_train_data)
        predicted_stress = model.predict(input_test_data)
        error=abs(testing[:,2]-predicted_stress)
        pError=100*error/testing[:,2]
        # Round to 2 decimal place
        temp = {'Slope':{leaveout_lot:testing[:,0].round(2)}, 'Strain':{leaveout_lot:testing[:,1].round(2)}, 'Stress':{leaveout_lot:testing[:,2].round(2)}, 'Predicted Stress':{leaveout_lot:predicted_stress.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info

def SVR_(data, leaveout_lot):  

        sys.stdout.flush()
        training = data.values[data.index!=leaveout_lot]
        testing = data.values[data.index==leaveout_lot]
        #verbose=0
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        #model = SVR(kernel='linear', C=100, gamma='auto')
        #model = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
        
        input_train_data = training[:,0:2]
        output_train_data = training[:,2]
        
        input_test_data=testing[:,0:2]
        #output_test_data=testing[:,2]
        model.fit(input_train_data, output_train_data)
        predicted_stress = model.predict(input_test_data)
        error=abs(testing[:,2]-predicted_stress)
        pError=100*error/testing[:,2]
        # Round to 2 decimal place
        temp = {'Slope':{leaveout_lot:testing[:,0].round(2)}, 'Strain':{leaveout_lot:testing[:,1].round(2)}, 'Stress':{leaveout_lot:testing[:,2].round(2)}, 'Predicted Stress':{leaveout_lot:predicted_stress.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info

def LG(data, leaveout_lot, degree):  

        sys.stdout.flush()
        training = data.values[data.index!=leaveout_lot]
        testing = data.values[data.index==leaveout_lot]
        #verbose=0
        poly = PolynomialFeatures(degree=degree)
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
    

def RF(data, leaveout_lot):

        sys.stdout.flush()
        training = data.values[data.index!=leaveout_lot]
        testing = data.values[data.index==leaveout_lot]

        verbose=0
        model = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)

        input_train_data = training[:,0:2]
        output_train_data = training[:,2]

        input_test_data=testing[:,0:2]
        model.fit(input_train_data, output_train_data)

        predicted_stress = model.predict(input_test_data)
        error=abs(testing[:,2]-predicted_stress)
        pError=100*error/testing[:,2]
        # Round to 2 decimal place
        temp = {'Slope':{leaveout_lot:testing[:,0].round(2)}, 'Strain':{leaveout_lot:testing[:,1].round(2)}, 'Stress':{leaveout_lot:testing[:,2].round(2)}, 'Predicted Stress':{leaveout_lot:predicted_stress.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
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

#RF
print("\nRandom-Forest:\n")
resultR=RF(dataset, dataset.index[0])
for i in dataset.index:
    temp=RF(dataset, i)
    resultR=resultR.append(temp)
resultR = resultR[1:dataset.index.size+1]
print(resultR)

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in resultR.index:
    MAPE+=resultR.loc[i,'%Error']
    RMSE+=(resultR.loc[i,'Absolute error'])**2
    MGT+=resultR.loc[i,'Stress']
MAPE/=resultR.index.size
RMSE/=resultR.index.size
RMSE=RMSE**(0.5)
MGT/=resultR.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

#LG
print("\nLinear Regression: 1\n")
resultL=LG(dataset, dataset.index[0],1)
for i in dataset.index:
    temp=LG(dataset, i,1)
    resultL=resultL.append(temp)
resultL = resultL[1:dataset.index.size+1]
print(resultL)

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in resultL.index:
    MAPE+=resultL.loc[i,'%Error']
    RMSE+=(resultL.loc[i,'Absolute error'])**2
    MGT+=resultL.loc[i,'Stress']
MAPE/=resultL.index.size
RMSE/=resultL.index.size
RMSE=RMSE**(0.5)
MGT/=resultL.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

#LG2
print("\nLinear Regression: 2\n")
resultL2=LG(dataset, dataset.index[0],2)
for i in dataset.index:
    temp=LG(dataset, i,2)
    resultL2=resultL2.append(temp)
resultL2 = resultL2[1:dataset.index.size+1]
print(resultL2)

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in resultL2.index:
    MAPE+=resultL2.loc[i,'%Error']
    RMSE+=(resultL2.loc[i,'Absolute error'])**2
    MGT+=resultL2.loc[i,'Stress']
MAPE/=resultL2.index.size
RMSE/=resultL2.index.size
RMSE=RMSE**(0.5)
MGT/=resultL2.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

#SVR
print("\nSupport Vector Regression:\n")
resultS=SVR_(dataset, dataset.index[0])
for i in dataset.index:
    temp=SVR_(dataset, i)
    resultS=resultS.append(temp)
resultS = resultS[1:dataset.index.size+1]
print(resultS)

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in resultS.index:
    MAPE+=resultS.loc[i,'%Error']
    RMSE+=(resultS.loc[i,'Absolute error'])**2
    MGT+=resultS.loc[i,'Stress']
MAPE/=resultS.index.size
RMSE/=resultS.index.size
RMSE=RMSE**(0.5)
MGT/=resultS.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

#GBR
print("\nGradient Boost Regression:\n")
resultG=GBR(dataset, dataset.index[0])
for i in dataset.index:
    temp=GBR(dataset, i)
    resultG=resultG.append(temp)
resultG = resultG[1:dataset.index.size+1]
print(resultG)

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in resultG.index:
    MAPE+=resultG.loc[i,'%Error']
    RMSE+=(resultG.loc[i,'Absolute error'])**2
    MGT+=resultG.loc[i,'Stress']
MAPE/=resultG.index.size
RMSE/=resultG.index.size
RMSE=RMSE**(0.5)
MGT/=resultG.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

#Plot/Compare
sort=resultR.sort_values(by="%Error")
plt.plot(sort.loc[:,'%Error'],'r-o', resultL.loc[sort.index,'%Error'],'b-o', resultL2.loc[sort.index,'%Error'],'m-o', resultS.loc[sort.index,'%Error'],'y-o', resultG.loc[sort.index,'%Error'],'g-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error')
plt.title('Prediction error: RF-L1-L2-SVR-GBR')
plt.legend()
plt.grid()
plt.show()
