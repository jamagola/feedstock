#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################
# Created on Thu Jun 13 12:32:07 2019
#
# @author: jaman1@llnl.gov
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

import warnings
warnings.filterwarnings("ignore")

def LR(data, leaveout_lot): 
   
        """
        Run a LR regression model on the data

        Arguments:
        data: The dataframe you will be using for your regression (pandas)
        leaveout_lot: The lot you will be leaving out (string)
       
        """

        sys.stdout.flush()
        training = data.values[data['lots']!=leaveout_lot]
        testing = data.values[data['lots']==leaveout_lot]

        X=np.vstack(training[:,0])
        input_train_data = X
        output_train_data = training[:,4]

        X_=np.vstack(testing[:,0])
        input_test_data=X_
        
        poly = PolynomialFeatures(degree=1)
        model = LinearRegression()
        
        input_train_data = poly.fit_transform(input_train_data)
        input_test_data = poly.fit_transform(input_test_data)
        
        model.fit(input_train_data, output_train_data)
        
        predicted_stress=0
        stress=0;
        count=0;
        for i in input_test_data:
            stress = model.predict(np.array([input_test_data[count]]))
            predicted_stress=np.append(predicted_stress,stress)
            count+=1
        predicted_stress=np.median(predicted_stress)
        error=abs(testing[0,4]-predicted_stress)
        pError=100*error/testing[0,4]
        # Round to 2 decimal place
        temp = {'Stress':{leaveout_lot:testing[0,4]}, 'Predicted Stress':{leaveout_lot:predicted_stress.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info

#SEM_output_data.pkl should be in working directory
#BSIF_df.pkl should be in working directory
pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
dataset=pd.DataFrame.from_dict(sem_data)

pickle_inB=open("BSIF_df.pkl","rb")
bsif_data=pickle.load(pickle_inB, encoding='latin1')
datasetB=pd.DataFrame.from_dict(bsif_data)

box=pd.DataFrame()
temp=pd.DataFrame()
result=pd.DataFrame()
temp0=pd.DataFrame()
        
for i in dataset.index:
    if (datasetB[datasetB['lots']==i].size)!=0 :    
        temp=datasetB[datasetB['lots']==i]
        temp['stress']=dataset['stress'][i]
        temp['strain']=dataset['strain'][i]
        temp['slope']=dataset['slope'][i]
        box=box.append(temp)
        del temp
        temp=pd.DataFrame()

lis=[]
for i in dataset.index:
    for j in box['lots']:
        if i==j:
            lis.append(i)
            break
        
for i in lis:
    temp0=LR(box, i)
    result=result.append(temp0)
    del temp0
    temp0=pd.DataFrame()
    print("Processed lot: {}\n".format(i))
    
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
plt.plot(sort.loc[:,'%Error'],'r-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error')
plt.title('Prediction error: LR')
plt.legend('R')
plt.grid()
plt.show()