#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:28:18 2019

@author: jaman1
"""

# This code generates predicted parameters from Human Assessed Feature(HAF) or BSIF and reuse 
# predicted parameters with HAF/BSIF to predict performance parameter.

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Ridge
#from sklearn import ensemble
from depFuse import *
import warnings
warnings.filterwarnings("ignore")

pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
dataset=pd.DataFrame.from_dict(sem_data)
label=pd.read_csv('humanN.csv')

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
"""
cnt=0
for i in datasetB['bsif']:
    datasetC['bsif'][cnt]=i[0:20]
    cnt+=1
"""
#datasetB=datasetC
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
        temp['s1']=label['stdev1'][n]
        temp['s2']=label['stdev2'][n]
        temp['s3']=label['stdev3'][n]
        temp['s4']=label['std4'][n]
        temp['s5']=label['std5'][n]
        box=box.append(temp)
        del temp
        temp=pd.DataFrame()
        n+=1


###################################################
for tl in label['lots']:

    labelR = label[label['lots']!=tl]   
    boxR = box[box['lots']!=tl]      
            
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
    
    ###############################################    
    for i in lis:
        temp0=fuse1(boxR,labelR, i) # Use labelR when not using BSIF
        result=result.append(temp0)
        del temp0
        temp0=pd.DataFrame()
        print('#',end=" ")
    
    #labelF=pd.read_csv('humanN.csv')
    ################# Attention ###################
    """
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
    """
    labelF = boxR
    labelF['~stress']=0.0
    labelF['~strain']=0.0
    labelF['~slope']=0.0
    
    for i in result.index:
        if i in labelF['lots'].values:
            labelF[labelF['lots']==i]['~stress']=result.loc[i]['Predicted Stress']
            labelF[labelF['lots']==i]['~strain']=result.loc[i]['Predicted Strain']
            labelF[labelF['lots']==i]['~slope']=result.loc[i]['Predicted Slope']
            
    modelF=fuseX1X(labelF)
    ########################TESTING
    del temp0
    temp0=pd.DataFrame()
    temp0=fuse1(box,label, tl)
    ########################
    #testF=pd.read_csv('humanN.csv')
    ################# Attention ###################
    """
    testF=testF[testF['lots']==tl]
    testF['~stress']=temp0['Predicted Stress'][tl]
    testF['~strain']=temp0['Predicted Strain'][tl]
    testF['~slope']=temp0['Predicted Slope'][tl]
    testF['stress']=temp0['Stress'][tl]
    testF['strain']=temp0['Strain'][tl]
    testF['slope']=temp0['Slope'][tl]
    """
    testF=box
    testF=testF[testF['lots']==tl]
    testF['~stress']=temp0['Predicted Stress'][tl]
    testF['~strain']=temp0['Predicted Strain'][tl]
    testF['~slope']=temp0['Predicted Slope'][tl]
    testing = testF.values
    cnt=0
    for i in testing[:,0]:
        testing[:,0][cnt]=np.append(i, testing[:,17:20][cnt])
        #testing[:,0][cnt][0:20]=np.append(i, testing[:,4:6][cnt])
        cnt+=1
    
    X_=np.vstack(testing[:,0])
    #X_=np.vstack(testing[:,0])[:,0:20]
    input_test_data=X_

    predicted_slope=0
    slope=0;
    count=0;
    for i in input_test_data:
        slope = modelF.predict(np.array([input_test_data[count]]))
        predicted_slope=np.append(predicted_slope,slope)
        count+=1
    predicted_slope=np.median(predicted_slope)
    error2=abs(testing[0,4]-predicted_slope)
    pError2=100*error2/testing[0,4]
    
    leaveout_lot=tl
    temp = {'Slope':{leaveout_lot:np.around(testing[0,4],2)},
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