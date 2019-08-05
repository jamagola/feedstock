#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################
# Created on Thu Jun 13 12:32:07 2019
#
# @author: jaman1@llnl.gov
###############################################################

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from pandas.plotting import scatter_matrix
from sklearn import ensemble
#import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")

def LR(data, leaveout_lot):  
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
        
        input_train_data = training[:,2] #Input: Stress
        output_train_data = training[:,0] #Output: Slope     
        input_train_data = poly.fit_transform(np.array([input_train_data]).reshape(-1,1))
        
        input_test_data = testing[:,2]
        input_test_data = poly.fit_transform(np.array([input_test_data]).reshape(-1,1))
        #output_test_data=testing[:,2]
 
        model.fit(input_train_data, output_train_data)
        predicted_out = model.predict(input_test_data)
        error=abs(testing[:,0]-predicted_out)
        pError=100*error/testing[:,0]
        # Round to 2 decimal place
        temp = {'Slope':{leaveout_lot:testing[:,0].round(2)}, 'Strain':{leaveout_lot:testing[:,1].round(2)}, 'Stress':{leaveout_lot:testing[:,2].round(2)}, 'Predicted Slope':{leaveout_lot:predicted_out.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info

def RF(data, leaveout_lot, model):
   
        """
        Run a RF regression model on the data

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
    
def RF_F(data, leaveout_lot, model):
   
        """
        Run a RF regression model on the data

        Arguments:
        data: The dataframe you will be using for your regression (pandas)
        leaveout_lot: The lot you will be leaving out (string)
       
        """

        sys.stdout.flush()
        training = data.values[data['lots']!=leaveout_lot]
        testing = data.values[data['lots']==leaveout_lot]
        
        cnt=0
        for i in training[:,0]:
            training[:,0][cnt]=np.append(i, training[:,7][cnt])
            cnt+=1
        
        cnt=0
        for i in testing[:,0]:
           testing[:,0][cnt]=np.append(i, testing[:,7][cnt])
           cnt+=1
        
        X=np.vstack(training[:,0])
        input_train_data = X
        output_train_data = training[:,6]

        X_=np.vstack(testing[:,0])
        input_test_data=X_
        model.fit(input_train_data, output_train_data) 
               
        predicted_slope=0
        slope=0;
        count=0;
        for i in input_test_data:
            slope = model.predict(np.array([input_test_data[count]]))
            predicted_slope=np.append(predicted_slope,slope)
            count+=1
        predicted_slope=np.median(predicted_slope)
        error=abs(testing[0,6]-predicted_slope)
        pError=100*error/testing[0,6]
        # Round to 2 decimal place
        temp = {'Slope':{leaveout_lot:testing[0,6]}, 'Predicted Slope':{leaveout_lot:predicted_slope.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)}} 
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
boxF=pd.DataFrame()
temp=pd.DataFrame()
result=pd.DataFrame()
resultF=pd.DataFrame()
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

verbose=0
model = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
modelF = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        
for i in lis:
    temp0=RF(box, i, model)
    result=result.append(temp0)
    del temp0
    temp0=pd.DataFrame()
    print("Processed lot: {}\n".format(i))
    
print(result)
################################################################################

for i in result.index:
    if (box[box['lots']==i].size)!=0 :    
        temp=box[box['lots']==i]
        temp['Predicted stress']=result['Predicted Stress'][i] #mean
        #temp['strain']=dataset['strain'][i]
        #temp['slope']=dataset['slope'][i]
        boxF=boxF.append(temp)
        del temp
        temp=pd.DataFrame()
        
for i in lis:
    temp0=RF_F(boxF, i, modelF)
    resultF=resultF.append(temp0)
    del temp0
    temp0=pd.DataFrame()
    print("Processed lot(fused): {}\n".format(i))
    
print(resultF)
################################################################################



MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in resultF.index:
    MAPE+=resultF.loc[i,'%Error']
    RMSE+=(resultF.loc[i,'Absolute error'])**2
    MGT+=resultF.loc[i,'Slope']
MAPE/=resultF.index.size
RMSE/=resultF.index.size
RMSE=RMSE**(0.5)
MGT/=resultF.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

#Plot/Compare
sort=resultF.sort_values(by="%Error")
plt.plot(sort.loc[:,'%Error'],'r-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error')
plt.title('Prediction error - Slope: RF+')
plt.legend('R')
plt.grid()
plt.show()