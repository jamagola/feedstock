#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:11:46 2019

@author: jaman1
"""

import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
    
def LG(data, leaveout_lot, err1, err2, mode):  
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
            input_train_data=training[:,[1,2]]+(1/100)*np.array([err1*training[:,1],err2*training[:,2]]).transpose()*(-1)**np.round(np.random.uniform(0,1,(np.size(training[:,2]),2)))
            input_test_data=testing[:,[1,2]]+(1/100)*np.array([err1*testing[:,1],err2*testing[:,2]]).transpose()*(-1)**np.round(np.random.uniform(0,1,(np.size(testing[:,2]),2)))
        else:
            input_train_data = training[:,[1,2]] #Input: Stress
            input_test_data=testing[:,[1,2]]+(1/100)*np.array([err1*testing[:,1],err2*testing[:,2]]).transpose()*(-1)**np.round(np.random.uniform(0,1,(np.size(testing[:,2]),2)))
            
        output_train_data = training[:,0] #Output: Slope         
        #input_train_data=input_train_data.reshape(-1,1)
        #input_test_data=input_test_data.reshape(-1,1)
        
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

step=10
measure=np.zeros([int(np.floor(101/step))+1,int(np.floor(101/step))+1])

for e1 in np.arange(0, 101, step):
    for e2 in np.arange(0, 101, step):
        result=pd.DataFrame()
        temp=pd.DataFrame()

        for i in dataset.index:
            temp=LG(dataset, i, e1, e2, True) ################################
            result=result.append(temp)
            del temp
            temp=pd.DataFrame()
            #print("Processed: Error1: {}% | Error2: {}%\n".format(e1, e2))

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
        measure[int(e1/step)][int(e2/step)]=MAPE

        del result
#print("\n\n")
#print("MAPE: {0}".format(np.around(MAPE,2)))
#print("RMSE: {0}".format(np.around(RMSE,2)))
#print("NRMSE: {0}".format(np.around(NRMSE,2)))
#print("\n\n")

#Plot/Compare
#sort=result.sort_values(by="%Error")
#plt.plot(err,measure, 'g-')
#plt.xlabel('%Error applied to LR')
#plt.ylabel('MAPE')
#plt.title('Trained: Truth - Predict: Error')
#plt.legend('G')
#plt.grid()
#plt.show()

#fig = plt.figure(figsize=(10.,10.))
#print('Error at the prediction end + training end')
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
er1 = np.arange(0, 101, step)
er2 = np.arange(0, 101, step)
er1, er2 = np.meshgrid(er1, er2) 

# Plot the surface.
surf = ax.plot_surface(er1, er2, measure, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 100)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('Strain error %')
ax.set_ylabel('Stress error %')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

