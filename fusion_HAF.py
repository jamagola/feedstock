#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:04:57 2019

@author: jaman1
"""

# This code generates predicted parameters from Human Assessed Feature(HAF) and reuse 
# predicted parameters with HAF to predict slope.

from depFusion_HAF import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
import warnings
warnings.filterwarnings("ignore")

#verbose=0
#model2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)

pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
dataset=pd.DataFrame.from_dict(sem_data)
label=pd.read_csv('humanN.csv')
info = pd.DataFrame()
label['stress']=0.0
label['strain']=0.0
label['slope']=0.0

n=0
for i in dataset.index:
    if i in label['lots'].values:
        label['stress'][n]=dataset.loc[i]['stress']
        label['strain'][n]=dataset.loc[i]['strain']
        label['slope'][n]=dataset.loc[i]['slope']
        n+=1
###################################################
for tl in label['lots']:

    labelR = label[label['lots']!=tl]       
            
    temp=pd.DataFrame()
    result=pd.DataFrame()
    resultF=pd.DataFrame()
    temp0=pd.DataFrame()
    
    lis=[]
    for i in dataset.index:
        for j in labelR['lots']:
            if i==j:
                lis.append(i)
                break
    
        
    for i in lis:
        temp0=fuse1(labelR, i)
        result=result.append(temp0)
        del temp0
        temp0=pd.DataFrame()
    
    labelF=pd.read_csv('humanN.csv')
    ################# Attention ###################
    labelF=labelF[labelF['lots']!=tl]
    labelF['~stress']=0.0
    labelF['~strain']=0.0
    labelF['~slope']=0.0
    labelF['stress']=0.0
    labelF['strain']=0.0
    labelF['slope']=0.0
    
    n=0
    for i in result.index:
        if i in labelF['lots'].values:
            labelF['~stress'][labelF.index[n]]=result.loc[i]['Predicted Stress']
            labelF['~strain'][labelF.index[n]]=result.loc[i]['Predicted Strain']
            labelF['~slope'][labelF.index[n]]=result.loc[i]['Predicted Slope']
            labelF['stress'][labelF.index[n]]=result.loc[i]['Stress']
            labelF['strain'][labelF.index[n]]=result.loc[i]['Strain']
            labelF['slope'][labelF.index[n]]=result.loc[i]['Slope']
            n+=1
            
    modelF=fuseX1X(labelF)
    ########################TESTING
    del temp0
    temp0=pd.DataFrame()
    temp0=fuse1(label, tl)
    ########################
    testF=pd.read_csv('humanN.csv')
    ################# Attention ###################
    testF=testF[testF['lots']==tl]
    testF['~stress']=temp0['Predicted Stress'][tl]
    testF['~strain']=temp0['Predicted Strain'][tl]
    testF['~slope']=temp0['Predicted Slope'][tl]
    testF['stress']=temp0['Stress'][tl]
    testF['strain']=temp0['Strain'][tl]
    testF['slope']=temp0['Slope'][tl]
    
    testing = testF.values
    X_=np.vstack(testing[:,11:14])
    input_test_data=X_
    
    predicted_slope=0
    predicted_slope = modelF.predict(input_test_data)
    error2=abs(testing[0,16]-predicted_slope)
    pError2=100*error2/testing[0,16]
    leaveout_lot=tl
    temp = {'Slope':{leaveout_lot:np.around(testing[0,16],2)},
            'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
            'Absolute error2':{leaveout_lot:np.around(error2,2)},
            '%Error2':{leaveout_lot:np.around(pError2,2)}} 
    info=info.append(pd.DataFrame.from_dict(temp))
    print('Processed lot: {}'.format(tl))
    
resultF=info    
print(resultF) 

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in resultF.index:
    MAPE+=resultF.loc[i,'%Error2']
    RMSE+=(resultF.loc[i,'Absolute error2'])**2
    MGT+=resultF.loc[i,'Slope']
MAPE/=resultF.index.size
RMSE/=resultF.index.size
RMSE=RMSE**(0.5)
MGT/=resultF.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(MAPE))
print("RMSE: {0}".format(RMSE))
print("NRMSE: {0}".format(NRMSE))
print("\n\n")

#Plot/Compare
sort=resultF.sort_values(by="%Error2")
plt.plot(sort.loc[:,'%Error2'],'b-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error')
plt.title('Prediction error: HAF+SS')
plt.legend('L')
plt.grid()
plt.show()