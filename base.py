#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:13:43 2019

@author: jaman1
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from baseDep import *
#from sklearn.linear_model import Ridge
from sklearn import ensemble
import warnings
warnings.filterwarnings("ignore")


##########################################
RF=True
##########################################
BSIF=True
HAF=True
INST=True


pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
dataset=pd.DataFrame.from_dict(sem_data)
label=pd.read_csv('complete.csv')

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

pickle_inB=open("BSIF_df.pkl","rb")
bsif_data=pickle.load(pickle_inB, encoding='latin1')
datasetB=pd.DataFrame.from_dict(bsif_data)
pickle_inA=open("bsif20.pkl","rb")
bsif_data2=pickle.load(pickle_inA, encoding='latin1')
datasetC=pd.DataFrame.from_dict(bsif_data2)

box=pd.DataFrame()
boxF=pd.DataFrame()
temp=pd.DataFrame()
result=pd.DataFrame()
resultF=pd.DataFrame()
temp0=pd.DataFrame()
info=pd.DataFrame()

n=0
for i in label['lots'].values:
    if (datasetB[datasetB['lots']==i].size)!=0 :    
        temp=datasetB[datasetB['lots']==i]
        temp['stress']=label['stress'][n]
        temp['strain']=label['strain'][n]
        temp['slope']=label['slope'][n]
        temp['Porosity']=label['porosity'][n]
        temp['Dispersity']=label['dispersity'][n]
        temp['Size']=label['size'][n]
        temp['Faceted']=label['facet'][n]
        temp['Area']=label['area'][n]
        temp['BET']=label['BET'][n]
        temp['15micron']=label['15micron'][n]
        temp['35micron']=label['35micron'][n]
        temp['60micron']=label['60micron'][n]
        temp['85micron']=label['85micron'][n]
        temp['105micron']=label['105micron'][n]
        temp['105M']=label['105M'][n]
        box=box.append(temp)
        del temp
        temp=pd.DataFrame()
        n+=1
lis=[]
for i in dataset.index:
    for j in box['lots']:
        if i==j:
            lis.append(i)
            break
##############################################################################
for i in lis:
    temp0=fuse1(box,label,i,RF, BSIF, HAF, INST)
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
plt.plot(sort.loc[:,'%Error0'],'b-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error-stress')
plt.title('Prediction error')
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
plt.plot(sort.loc[:,'%Error1'],'g-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error-strain')
plt.title('Prediction error')
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
plt.plot(sort.loc[:,'%Error2'],'r-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error-slope')
plt.title('Prediction error')
plt.grid()
plt.show()

result.to_csv('/Users/jaman1/Desktop/bsif_lr.csv',index=True)