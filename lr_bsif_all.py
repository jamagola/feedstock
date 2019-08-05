#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:15:46 2019

@author: jaman1
"""

import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
        
        model = LinearRegression() 
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
        
        
        X=np.vstack(training[:,0])
        input_train_data = X
        output_train_data = training[:,5]

        X_=np.vstack(testing[:,0])
        input_test_data=X_
        
        model2 = LinearRegression() 
        model2.fit(input_train_data, output_train_data)
        
        predicted_strain=0
        strain=0;
        count=0;
        for i in input_test_data:
            strain = model2.predict(np.array([input_test_data[count]]))
            predicted_strain=np.append(predicted_strain,strain)
            count+=1
        predicted_strain=np.median(predicted_strain)
        error2=abs(testing[0,5]-predicted_strain)
        pError2=100*error2/testing[0,5]
        
        X=np.vstack(training[:,0])
        input_train_data = X
        output_train_data = training[:,6]

        X_=np.vstack(testing[:,0])
        input_test_data=X_
        
        model3 = LinearRegression() 
        model3.fit(input_train_data, output_train_data)
        
        predicted_slope=0
        slope=0;
        count=0;
        for i in input_test_data:
            slope = model3.predict(np.array([input_test_data[count]]))
            predicted_slope=np.append(predicted_slope,slope)
            count+=1
        predicted_slope=np.median(predicted_slope)
        error3=abs(testing[0,6]-predicted_slope)
        pError3=100*error3/testing[0,6]
        
        # Round to 2 decimal place
        temp = {'Stress':{leaveout_lot:testing[0,4]},'Strain':{leaveout_lot:testing[0,5]},'Slope':{leaveout_lot:testing[0,6]}, 'Predicted Stress':{leaveout_lot:predicted_stress.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)},'Predicted Strain':{leaveout_lot:predicted_strain.round(2)}, 'Absolute error2':{leaveout_lot:error2.round(2)}, '%Error2':{leaveout_lot:pError2.round(2)}, 'Predicted Slope':{leaveout_lot:predicted_slope.round(2)}, 'Absolute error3':{leaveout_lot:error3.round(2)}, '%Error3':{leaveout_lot:pError3.round(2)}} 
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
print('Status: Stress')
for i in result.index:
    MAPE+=result.loc[i,'%Error']
    RMSE+=(result.loc[i,'Absolute error'])**2
    MGT+=result.loc[i,'Stress']
MAPE/=result.index.size
RMSE/=result.index.size
RMSE=RMSE**(0.5)
MGT/=result.index.size
NRMSE=RMSE/MGT
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

MAPE=0
RMSE=0
MGT=0
print('Status: Strain')
for i in result.index:
    MAPE+=result.loc[i,'%Error2']
    RMSE+=(result.loc[i,'Absolute error2'])**2
    MGT+=result.loc[i,'Strain']
MAPE/=result.index.size
RMSE/=result.index.size
RMSE=RMSE**(0.5)
MGT/=result.index.size
NRMSE=RMSE/MGT
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

MAPE=0
RMSE=0
MGT=0
print('Status: Slope')
for i in result.index:
    MAPE+=result.loc[i,'%Error3']
    RMSE+=(result.loc[i,'Absolute error3'])**2
    MGT+=result.loc[i,'Slope']
MAPE/=result.index.size
RMSE/=result.index.size
RMSE=RMSE**(0.5)
MGT/=result.index.size
NRMSE=RMSE/MGT
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

#Plot/Compare
sort=result.sort_values(by="%Error")
plt.plot(sort.loc[:,'%Error'],'r-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error')
plt.title('Prediction error: LR-Stress')
plt.legend('L')
plt.grid()
plt.show()

sort=result.sort_values(by="%Error2")
plt.plot(sort.loc[:,'%Error2'],'g-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error')
plt.title('Prediction error: LR-Strain')
plt.legend('L')
plt.grid()
plt.show()

sort=result.sort_values(by="%Error3")
plt.plot(sort.loc[:,'%Error3'],'b-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error')
plt.title('Prediction error: LR-Slope')
plt.legend('L')
plt.grid()
plt.show()