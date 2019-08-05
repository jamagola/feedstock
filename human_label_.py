#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:31:03 2019

@author: jaman1
"""

import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
#import os
import sys
import pandas as pd 
import numpy as np
#import scipy
import warnings
warnings.filterwarnings("ignore")

def RL(data, leaveout_lot):  
        """
        Run a Linear regression model on the data

        Arguments:
        data: The dataframe you will be using for your regression (pandas)
        leaveout_lot: The lot you will be leaving out (string)
       
        """
        sys.stdout.flush()
        training = data.values[data['Lot'].values!=leaveout_lot]
        testing = data.values[data['Lot'].values==leaveout_lot]

        model0 = LinearRegression()
        model1 = LinearRegression()
        model2 = LinearRegression()
        
        #verbose=0
        #model0 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        #model1 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        #model2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        
        input_train_data = training[:,1:5]
        output_train_data0 = training[:,5]   
        output_train_data1 = training[:,6] 
        output_train_data2 = training[:,7] 
        
        input_test_data = testing[:,1:5]
        output_test_data0=testing[:,5]
        output_test_data1=testing[:,6]
        output_test_data2=testing[:,7]
 
        model0.fit(input_train_data, output_train_data0)
        model1.fit(input_train_data, output_train_data1)
        model2.fit(input_train_data, output_train_data2)
    
        predicted_stress = model0.predict(input_test_data)
        error0=abs(output_test_data0-predicted_stress)
        pError0=100*error0/output_test_data0
        
        predicted_strain = model1.predict(input_test_data)
        error1=abs(output_test_data1-predicted_strain)
        pError1=100*error1/output_test_data1
        
        predicted_slope = model2.predict(input_test_data)
        error2=abs(output_test_data2-predicted_slope)
        pError2=100*error2/output_test_data2
        
        # Round to 2 decimal place
        temp = {'Stress':{leaveout_lot:np.around(testing[:,5].astype(np.double),2)},
                'Predicted Stress':{leaveout_lot:np.around(predicted_stress.astype(np.double),2)},
                'Absolute error0':{leaveout_lot:np.around(error0.astype(np.double),2)}, 
                '%Error0':{leaveout_lot:np.around(pError0.astype(np.double),2)},
                'Strain':{leaveout_lot:np.around(testing[:,6].astype(np.double),2)},
                'Predicted Strain':{leaveout_lot:np.around(predicted_strain.astype(np.double),2)},
                'Absolute error1':{leaveout_lot:np.around(error1.astype(np.double),2)}, 
                '%Error1':{leaveout_lot:np.around(pError1.astype(np.double),2)},
                'Slope':{leaveout_lot:np.around(testing[:,7].astype(np.double),2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope.astype(np.double),2)},
                'Absolute error2':{leaveout_lot:np.around(error2.astype(np.double),2)},
                '%Error2':{leaveout_lot:np.around(pError2.astype(np.double),2)}} 
      
        info=pd.DataFrame.from_dict(temp)
        return info

#SEM_output_data.pkl should be in working directory
pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
dataset=pd.DataFrame.from_dict(sem_data)
label=pd.read_csv('human.csv')
#print("\nPrinting generic statistics: \n")
#print(dataset.describe())
#print("\n\n")

label['stress']=0.0
label['strain']=0.0
label['slope']=0.0

n=0
for i in dataset.index:
    if i in label['Lot'].values:
        label['stress'][n]=dataset.loc[i]['stress']
        label['strain'][n]=dataset.loc[i]['strain']
        label['slope'][n]=dataset.loc[i]['slope']
        n+=1

result=pd.DataFrame()
for i in label['Lot'].values:
    temp=RL(label, i)
    result=result.append(temp)
print(result)

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in result.index:
    MAPE+=result.loc[i,'%Error0']
    RMSE+=(result.loc[i,'Absolute error0'])**2
    MGT+=result.loc[i,'Stress']
MAPE/=result.index.size
RMSE/=result.index.size
RMSE=RMSE**(0.5)
MGT/=result.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(MAPE))
print("RMSE: {0}".format(RMSE))
print("NRMSE: {0}".format(NRMSE))
print("\n\n")

#Plot/Compare
sort=result.sort_values(by="%Error0")
plt.plot(sort.loc[:,'%Error0'],'g-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error-stress')
plt.title('Prediction error: LR')
plt.legend('L')
plt.grid()
plt.show()

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in result.index:
    MAPE+=result.loc[i,'%Error1']
    RMSE+=(result.loc[i,'Absolute error1'])**2
    MGT+=result.loc[i,'Strain']
MAPE/=result.index.size
RMSE/=result.index.size
RMSE=RMSE**(0.5)
MGT/=result.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(MAPE))
print("RMSE: {0}".format(RMSE))
print("NRMSE: {0}".format(NRMSE))
print("\n\n")

#Plot/Compare
sort=result.sort_values(by="%Error1")
plt.plot(sort.loc[:,'%Error1'],'r-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error-strain')
plt.title('Prediction error: LR')
plt.legend('L')
plt.grid()
plt.show()

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in result.index:
    MAPE+=result.loc[i,'%Error2']
    RMSE+=(result.loc[i,'Absolute error2'])**2
    MGT+=result.loc[i,'Slope']
MAPE/=result.index.size
RMSE/=result.index.size
RMSE=RMSE**(0.5)
MGT/=result.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(MAPE))
print("RMSE: {0}".format(RMSE))
print("NRMSE: {0}".format(NRMSE))
print("\n\n")

#Plot/Compare
sort=result.sort_values(by="%Error2")
plt.plot(sort.loc[:,'%Error2'],'b-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error-slope')
plt.title('Prediction error: LR')
plt.legend('L')
plt.grid()
plt.show()