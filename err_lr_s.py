#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:07:22 2019

@author: jaman1
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
    
def LG(data, leaveout_lot, err, mode):  
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
        model=LinearRegression()
        
        if mode==True:
            input_train_data=training[:,2]+(err/100)*training[:,2]*(-1)**np.round(np.random.uniform(0,1,np.size(training[:,2])))
            input_test_data=testing[:,2]+(err/100)*testing[:,2]*(-1)**np.round(np.random.uniform(0,1,np.size(testing[:,2])))
        else:
            input_train_data = training[:,2] #Input: Stress
            input_test_data=testing[:,2]+(err/100)*testing[:,2]*(-1)**np.round(np.random.uniform(0,1,np.size(testing[:,2])))
            
        output_train_data = training[:,0] #Output: Slope         
        input_train_data=input_train_data.reshape(-1,1)
        input_test_data=input_test_data.reshape(-1,1)
        
        model.fit(input_train_data, output_train_data)
        predicted_slope = model.predict(input_test_data)
        
        error=abs(testing[:,0]-predicted_slope)
        pError=100*error/testing[:,0]
        # Round to 2 decimal place
        temp = {'Slope':{leaveout_lot:testing[:,0]}, 'Predicted Slope':{leaveout_lot:predicted_slope.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info
    
#SEM_output_data.pkl should be in working directory
#BSIF_df.pkl should be in working directory
pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
dataset=pd.DataFrame.from_dict(sem_data)

measure=np.array([])
err=np.array([])

for e in range(0,101):
    result=pd.DataFrame()
    temp=pd.DataFrame()

    for i in dataset.index:
        temp=LG(dataset, i, e, True) #########################################
        result=result.append(temp)
        del temp
        temp=pd.DataFrame()
        #print("Processed lot: {}\n".format(i))

    #print(result)

    MAPE=0
    RMSE=0
    MGT=0
    NRMSE=0
    # Measuring error index
    for i in result.index:
        MAPE+=result.loc[i,'%Error']
        RMSE+=(result.loc[i,'Absolute error'])**2
        MGT+=result.loc[i,'Slope']
    MAPE/=result.index.size
    RMSE/=result.index.size
    RMSE=RMSE**(0.5)
    MGT/=result.index.size
    NRMSE=RMSE/MGT
    measure=np.append(measure,MAPE)
    err=np.append(err, e)
    del result
#print("\n\n")
#print("MAPE: {0}".format(np.around(MAPE,2)))
#print("RMSE: {0}".format(np.around(RMSE,2)))
#print("NRMSE: {0}".format(np.around(NRMSE,2)))
#print("\n\n")

#Plot/Compare
#sort=result.sort_values(by="%Error")
plt.plot(err,measure, 'g-')
plt.xlabel('%Error applied to LR')
plt.ylabel('MAPE')
plt.title('Trained: Error - Predict: Error')
#plt.legend('G')
plt.grid()
plt.show()

