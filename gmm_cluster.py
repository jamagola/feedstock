#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:59:13 2019

@author: jaman1
"""

import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import sys
import pandas as pd 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")
   
def gmmx(data, leaveout_lot, c): 

        sys.stdout.flush()
        training = data.values[data['lots']!=leaveout_lot]
        testing = data.values[data['lots']==leaveout_lot]

        X=np.vstack(training[:,0])
        input_train_data = X
        #output_train_data = training[0:100,?]

        X_=np.vstack(testing[0:100,0]) ########################
        input_test_data=X_
        
        model = GaussianMixture(n_components=c) ################
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
        temp=datasetB[datasetB['lots']==i][0:100] #####################
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
for cl in np.arange(10,15,5):
    for i in lis:
        temp0=gmmx(box, i,cl)
        result=result.append(temp0)
        del temp0
        temp0=pd.DataFrame()
        #print('#', end='')
        #print("Processed lot: {}\n".format(i))
        
        plt.figure()
        plt.plot(result.loc[i][0])
        plt.xlabel('Cluster')
        plt.ylabel('Distribution')
        plt.title(i)
        plt.grid()
        plt.show    
        
        
    print('\n')
    #print(result)
    
    
    measure=np.array([result.loc['AM'][0]])
    for i in lis:
        measure = np.concatenate((measure, np.array([result.loc[i][0]])), axis=0)
    measure=np.delete(measure, (0), axis=0)
    
    l = np.arange(0,len(result),1)
    c=np.arange(0, cl, 1) ##############
    c,l=np.meshgrid(c, l)
    
    cp = plt.contourf(c, l, measure)
    plt.colorbar(cp)
    plt.title('Cluster: {0}'.format(cl)) #######
    plt.xlabel('cluster')
    plt.ylabel('lots')
    plt.show()
    del result
    result=pd.DataFrame()