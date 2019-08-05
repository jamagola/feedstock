#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:59:13 2019

@author: jaman1
"""

import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
#import os
import sys
import pandas as pd 
import numpy as np
#import scipy

import warnings
warnings.filterwarnings("ignore")
   
def gmmx(data, leaveout_lot): 
   
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
        #output_train_data = training[0:100,?]

        X_=np.vstack(testing[0:300,0]) ########################
        input_test_data=X_
        
        #XY=np.concatenate((input_train_data,output_train_data.reshape(-1,1)),axis=-1)
        #test=np.concatenate((input_test_data,testing[:,7].reshape(-1,1)),axis=-1)
        
        model = GaussianMixture(n_components=10)
        model.fit(X)
        
        predicted_class=[]
        clss=[]
        count=0
        predicted_class=model.predict_proba(np.array([X_[count]]))
        for i in input_test_data:
            clss=model.predict_proba(np.array([X_[count]]))
            predicted_class=np.concatenate((predicted_class, clss), axis=0)
            count+=1
        predicted_class = np.delete(predicted_class, (0), axis=0)
        predicted_class=predicted_class.mean(axis=0)
        #print(predicted_class)
        
        # Round to 2 decimal place
        temp = {'Predicted Class':{leaveout_lot:predicted_class}} 
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
resultF=pd.DataFrame()

num=0

for i in dataset.index:
    if (datasetB[datasetB['lots']==i].size)!=0 :    
        temp=datasetB[datasetB['lots']==i][0:300] #####################
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

##################################################################################            
"""
from multiprocessing import Process, Manager
 
manager = Manager()
predictions = manager.dict()
 
procs = []
 
for i in lis:
    proc = Process(target=KNNX, args=(box, i))
    procs.append(proc)
    temp0=proc.start()
    result=result.append(temp0)
    del temp0
    temp0=pd.DataFrame()
 
for proc in procs:
    proc.join()            

"""    
###################################################################################
for i in lis:
    temp0=gmmx(box, i)
    result=result.append(temp0)
    del temp0
    temp0=pd.DataFrame()
    print("Processed lot: {}\n".format(i))
    
print(result)

for i in lis:
    plt.figure()
    plt.plot(result.loc[i][0])
    plt.xlabel('Cluster')
    plt.ylabel('Distribution')
    plt.title(i)
    plt.grid()
    plt.show

measure=np.array([result.loc['AM'][0]])
for i in lis:
    measure = np.concatenate((measure, np.array([result.loc[i][0]])), axis=0)
measure=np.delete(measure, (0), axis=0)

from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
#l=result.index
l = np.arange(len(result))
c=np.arange(0, 10, 1)
c,l=np.meshgrid(c, l)

# Plot the surface.
surf = ax.plot_surface(c, l, measure, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(0.1))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_ylabel('Lots')
ax.set_xlabel('Cluster')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()