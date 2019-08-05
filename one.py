#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:10:50 2019

@author: jaman1
"""
#0: BSIF 1:HAF 2:BSIF+SS 3:HAF+SS 4:HAF+SS(RF) 5:HAF(RF)
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
import warnings
warnings.filterwarnings("ignore")
mode = 0

def bsif_lr(data, leaveout_lot):

        model2=LinearRegression()
        M=5
        sys.stdout.flush()
        training = data.values[data['lots']!=leaveout_lot]
        testing = data.values[data['lots']==leaveout_lot]
        
        X=np.vstack(training[:,0])
        input_train_data = X
        output_train_data2 = training[:,M]
        X_=np.vstack(testing[:,0])
        input_test_data=X_
        
        model2.fit(input_train_data, output_train_data2)
        predicted_slope=0
        slope=0;
        count=0;
        for i in input_test_data:
            slope = model2.predict(np.array([input_test_data[count]]))
            predicted_slope=np.append(predicted_slope,slope)
            count+=1
        predicted_slope=np.median(predicted_slope)
        error2=abs(testing[0,M]-predicted_slope)
        pError2=100*error2/testing[0,M]
        ##################################################################
        
        # Round to 2 decimal place        
        temp = {'Slope':{leaveout_lot:np.around(testing[0,M],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        
        info=pd.DataFrame.from_dict(temp)
        return info

def haf(data, leaveout_lot):

        model2=LinearRegression()

        sys.stdout.flush()
        training = data.values[data['Lot']!=leaveout_lot]
        testing = data.values[data['Lot']==leaveout_lot]
        
        X=np.vstack(training[:,1:5])
        input_train_data = X
        output_train_data2 = training[:,7]
        X_=np.vstack(testing[:,1:5])
        input_test_data=X_
        
        model2.fit(input_train_data, output_train_data2)
        predicted_slope= model2.predict(np.array(input_test_data))
        error2=abs(testing[0,7]-predicted_slope)
        pError2=100*error2/testing[0,7]
    
        temp = {'Slope':{leaveout_lot:np.around(testing[0,7],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        
        info=pd.DataFrame.from_dict(temp)
        return info
    
def haf2(data, leaveout_lot):

        verbose=0
        model2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)

        sys.stdout.flush()
        training = data.values[data['Lot']!=leaveout_lot]
        testing = data.values[data['Lot']==leaveout_lot]
        
        X=np.vstack(training[:,1:5])
        input_train_data = X
        output_train_data2 = training[:,7]
        X_=np.vstack(testing[:,1:5])
        input_test_data=X_
        
        model2.fit(input_train_data, output_train_data2)
        predicted_slope= model2.predict(np.array(input_test_data))
        error2=abs(testing[0,7]-predicted_slope)
        pError2=100*error2/testing[0,7]
    
        temp = {'Slope':{leaveout_lot:np.around(testing[0,7],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        
        info=pd.DataFrame.from_dict(temp)
        return info

def fuse0(data, leaveout_lot):

        model0=LinearRegression()
        model1=LinearRegression()
        model2=LinearRegression()

        sys.stdout.flush()
        training = data.values[data['lots']!=leaveout_lot]
        testing = data.values[data['lots']==leaveout_lot]
        
        X=np.vstack(training[:,0])
        input_train_data = X
        output_train_data0 = training[:,4]
        output_train_data1 = training[:,5]
        output_train_data2 = training[:,6]
        X_=np.vstack(testing[:,0])
        input_test_data=X_
        
        model0.fit(input_train_data, output_train_data0)
        model1.fit(input_train_data, output_train_data1)
        model2.fit(input_train_data, output_train_data2)
        
        ##################################################################
        predicted_stress=0
        stress=0;
        count=0;
        for i in input_test_data:
            stress = model0.predict(np.array([input_test_data[count]]))
            predicted_stress=np.append(predicted_stress,stress)
            count+=1
        predicted_stress=np.median(predicted_stress)
        error0=abs(testing[0,4]-predicted_stress)
        pError0=100*error0/testing[0,4]
        
        ##################################################################
        predicted_strain=0
        strain=0;
        count=0;
        for i in input_test_data:
            strain = model1.predict(np.array([input_test_data[count]]))
            predicted_strain=np.append(predicted_strain,strain)
            count+=1
        predicted_strain=np.median(predicted_strain)
        error1=abs(testing[0,5]-predicted_strain)
        pError1=100*error1/testing[0,5]
        
        ##################################################################
        predicted_slope=0
        slope=0;
        count=0;
        for i in input_test_data:
            slope = model2.predict(np.array([input_test_data[count]]))
            predicted_slope=np.append(predicted_slope,slope)
            count+=1
        predicted_slope=np.median(predicted_slope)
        error2=abs(testing[0,6]-predicted_slope)
        pError2=100*error2/testing[0,6]
        ##################################################################
        
        # Round to 2 decimal place        
        temp = {'Stress':{leaveout_lot:np.around(testing[0,4],2)},
                'Predicted Stress':{leaveout_lot:np.around(predicted_stress,2)},
                'Absolute error0':{leaveout_lot:np.around(error0,2)}, 
                '%Error0':{leaveout_lot:np.around(pError0,2)},
                'Strain':{leaveout_lot:np.around(testing[0,5],2)},
                'Predicted Strain':{leaveout_lot:np.around(predicted_strain,2)},
                'Absolute error1':{leaveout_lot:np.around(error1,2)}, 
                '%Error1':{leaveout_lot:np.around(pError1,2)},
                'Slope':{leaveout_lot:np.around(testing[0,6],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        
        info=pd.DataFrame.from_dict(temp)
        return info
    
def fuse1(data, leaveout_lot):

        model0=LinearRegression()
        model1=LinearRegression()
        model2=LinearRegression()
        sys.stdout.flush()
        training = data.values[data['Lot']!=leaveout_lot]
        testing = data.values[data['Lot']==leaveout_lot]

        X=np.vstack(training[:,1:5])
        input_train_data = X
        output_train_data0 = training[:,5]
        output_train_data1 = training[:,6]
        output_train_data2 = training[:,7]
        X_=np.vstack(testing[:,1:5])
        input_test_data=X_
        
        model0.fit(input_train_data, output_train_data0)
        model1.fit(input_train_data, output_train_data1)
        model2.fit(input_train_data, output_train_data2)
        
        ##################################################################
        predicted_stress=0
        predicted_stress = model0.predict(input_test_data)
        error0=abs(testing[0,5]-predicted_stress)
        pError0=100*error0/testing[0,5]
        
        ##################################################################
        predicted_strain=0
        predicted_strain = model1.predict(input_test_data)
        error1=abs(testing[0,6]-predicted_strain)
        pError1=100*error1/testing[0,6]
        
        ##################################################################
        predicted_slope=0
        predicted_slope = model2.predict(input_test_data)
        error2=abs(testing[0,7]-predicted_slope)
        pError2=100*error2/testing[0,7]
        ##################################################################
        
        # Round to 2 decimal place        
        temp = {'Stress':{leaveout_lot:np.around(testing[0,5],2)},
                'Predicted Stress':{leaveout_lot:np.around(predicted_stress,2)},
                'Absolute error0':{leaveout_lot:np.around(error0,2)}, 
                '%Error0':{leaveout_lot:np.around(pError0,2)},
                'Strain':{leaveout_lot:np.around(testing[0,6],2)},
                'Predicted Strain':{leaveout_lot:np.around(predicted_strain,2)},
                'Absolute error1':{leaveout_lot:np.around(error1,2)}, 
                '%Error1':{leaveout_lot:np.around(pError1,2)},
                'Slope':{leaveout_lot:np.around(testing[0,7],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        
        info=pd.DataFrame.from_dict(temp)
        return info
    
def fuse2(data, leaveout_lot):

        verbose=0
        model0 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        model1 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        model2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        sys.stdout.flush()
        training = data.values[data['Lot']!=leaveout_lot]
        testing = data.values[data['Lot']==leaveout_lot]

        X=np.vstack(training[:,1:5])
        input_train_data = X
        output_train_data0 = training[:,5]
        output_train_data1 = training[:,6]
        output_train_data2 = training[:,7]
        X_=np.vstack(testing[:,1:5])
        input_test_data=X_
        
        model0.fit(input_train_data, output_train_data0)
        model1.fit(input_train_data, output_train_data1)
        model2.fit(input_train_data, output_train_data2)
        
        ##################################################################
        predicted_stress=0
        predicted_stress = model0.predict(input_test_data)
        error0=abs(testing[0,5]-predicted_stress)
        pError0=100*error0/testing[0,5]
        
        ##################################################################
        predicted_strain=0
        predicted_strain = model1.predict(input_test_data)
        error1=abs(testing[0,6]-predicted_strain)
        pError1=100*error1/testing[0,6]
        
        ##################################################################
        predicted_slope=0
        predicted_slope = model2.predict(input_test_data)
        error2=abs(testing[0,7]-predicted_slope)
        pError2=100*error2/testing[0,7]
        ##################################################################
        
        # Round to 2 decimal place        
        temp = {'Stress':{leaveout_lot:np.around(testing[0,5],2)},
                'Predicted Stress':{leaveout_lot:np.around(predicted_stress,2)},
                'Absolute error0':{leaveout_lot:np.around(error0,2)}, 
                '%Error0':{leaveout_lot:np.around(pError0,2)},
                'Strain':{leaveout_lot:np.around(testing[0,6],2)},
                'Predicted Strain':{leaveout_lot:np.around(predicted_strain,2)},
                'Absolute error1':{leaveout_lot:np.around(error1,2)}, 
                '%Error1':{leaveout_lot:np.around(pError1,2)},
                'Slope':{leaveout_lot:np.around(testing[0,7],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        
        info=pd.DataFrame.from_dict(temp)
        return info
    
def fuseX0(data, leaveout_lot):

        model2=LinearRegression()
        sys.stdout.flush()
        training = data.values[data['lots']!=leaveout_lot]
        testing = data.values[data['lots']==leaveout_lot]
        
        cnt=0
        for i in training[:,0]:
            training[:,0][cnt]=np.append(i, training[:,4:6][cnt])
            cnt+=1
        
        cnt=0
        for i in testing[:,0]:
           testing[:,0][cnt]=np.append(i, testing[:,4:6][cnt])
           cnt+=1
        
        X=np.vstack(training[:,0])
        input_train_data = X
        output_train_data2 = training[:,13]
        X_=np.vstack(testing[:,0])
        input_test_data=X_

        model2.fit(input_train_data, output_train_data2)
 
        predicted_slope=0
        slope=0;
        count=0;
        for i in input_test_data:
            slope = model2.predict(np.array([input_test_data[count]]))
            predicted_slope=np.append(predicted_slope,slope)
            count+=1
        predicted_slope=np.median(predicted_slope)
        error2=abs(testing[0,13]-predicted_slope)
        pError2=100*error2/testing[0,13]
              
        temp = {'Slope':{leaveout_lot:np.around(testing[0,13],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        
        info=pd.DataFrame.from_dict(temp)
        return info
    
def fuseX1(data, leaveout_lot):
     
        model2=LinearRegression()

        sys.stdout.flush()
        training = data.values[data['Lot']!=leaveout_lot]
        testing = data.values[data['Lot']==leaveout_lot]

        X=np.vstack(training[:,1:7])
        input_train_data = X
        output_train_data2 = training[:,10]
        X_=np.vstack(testing[:,1:7])
        input_test_data=X_

        model2.fit(input_train_data, output_train_data2)

        predicted_slope=0
        predicted_slope = model2.predict(input_test_data)
        error2=abs(testing[0,10]-predicted_slope)
        pError2=100*error2/testing[0,10]
        ##################################################################
        
        # Round to 2 decimal place 
        temp = {'Slope':{leaveout_lot:np.around(testing[0,10],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info
    
def fuseX2(data, leaveout_lot):
     
        verbose=0
        model2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)

        sys.stdout.flush()
        training = data.values[data['Lot']!=leaveout_lot]
        testing = data.values[data['Lot']==leaveout_lot]

        X=np.vstack(training[:,1:7])
        input_train_data = X
        output_train_data2 = training[:,10]
        X_=np.vstack(testing[:,1:7])
        input_test_data=X_

        model2.fit(input_train_data, output_train_data2)

        predicted_slope=0
        predicted_slope = model2.predict(input_test_data)
        error2=abs(testing[0,10]-predicted_slope)
        pError2=100*error2/testing[0,10]
        ##################################################################
        
        # Round to 2 decimal place 
        temp = {'Slope':{leaveout_lot:np.around(testing[0,10],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info
    
pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
dataset=pd.DataFrame.from_dict(sem_data)
label=pd.read_csv('human.csv')

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

pickle_inB=open("BSIF_df.pkl","rb")
bsif_data=pickle.load(pickle_inB, encoding='latin1')
datasetB=pd.DataFrame.from_dict(bsif_data)

box=pd.DataFrame()
boxF=pd.DataFrame()
temp=pd.DataFrame()
result=pd.DataFrame()
resultF=pd.DataFrame()
temp0=pd.DataFrame()

n=0
for i in label['Lot'].values:
    if (datasetB[datasetB['lots']==i].size)!=0 :    
        temp=datasetB[datasetB['lots']==i]
        temp['stress']=label['stress'][n]
        temp['strain']=label['strain'][n]
        temp['slope']=label['slope'][n]
        temp['Porosity']=label['Porosity'][n]
        temp['Dispersity']=label['Dispersity'][n]
        temp['Size']=label['Size'][n]
        temp['Faceted']=label['Faceted'][n]
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

if mode==0:            
    for i in lis:
        temp0=bsif_lr(box, i)
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
    plt.title('Prediction error: BSIF')
    plt.legend('L')
    plt.grid()
    plt.show()


if mode==1:
    for i in lis:
        temp0=haf(label, i)
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
    plt.title('Prediction error: HAF')
    plt.legend('L')
    plt.grid()
    plt.show()
    

if mode==2:
    for i in lis:
        temp0=fuse0(box, i)
        result=result.append(temp0)
        del temp0
        temp0=pd.DataFrame()
        print("Processed lot: {}\n".format(i))
    
    labelF=pd.read_csv('human.csv')
    labelF['~stress']=0.0
    labelF['~strain']=0.0
    labelF['~slope']=0.0
    labelF['stress']=0.0
    labelF['strain']=0.0
    labelF['slope']=0.0
    
    n=0
    for i in result.index:
        if i in labelF['Lot'].values:
            labelF['~stress'][n]=result.loc[i]['Predicted Stress']
            labelF['~strain'][n]=result.loc[i]['Predicted Strain']
            labelF['~slope'][n]=result.loc[i]['Predicted Slope']
            labelF['stress'][n]=result.loc[i]['Stress']
            labelF['strain'][n]=result.loc[i]['Strain']
            labelF['slope'][n]=result.loc[i]['Slope']
            n+=1
            
    n=0
    for i in labelF['Lot'].values:
        if (datasetB[datasetB['lots']==i].size)!=0 :    
            temp=datasetB[datasetB['lots']==i]
            temp['~stress']=labelF['~stress'][n]
            temp['~strain']=labelF['~strain'][n]
            temp['~slope']=labelF['~slope'][n]
            temp['Porosity']=label['Porosity'][n]
            temp['Dispersity']=label['Dispersity'][n]
            temp['Size']=label['Size'][n]
            temp['Faceted']=label['Faceted'][n]
            temp['stress']=labelF['stress'][n]
            temp['strain']=labelF['strain'][n]
            temp['slope']=labelF['slope'][n]
            boxF=boxF.append(temp)
            del temp
            temp=pd.DataFrame()
            n+=1
            
    for i in lis:
        temp0=fuseX0(boxF, i)
        resultF=resultF.append(temp0)
        del temp0
        temp0=pd.DataFrame()
        print("Processed lot(fused): {}\n".format(i))
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
    plt.ylabel('%Error-slope')
    plt.title('Prediction error: BSIF+SS')
    plt.legend('L')
    plt.grid()
    plt.show()
    
    
    
if mode==3:
    for i in lis:
        temp0=fuse1(label, i)
        result=result.append(temp0)
        del temp0
        temp0=pd.DataFrame()
    
    labelF=pd.read_csv('human.csv')
    labelF['~stress']=0.0
    labelF['~strain']=0.0
    labelF['~slope']=0.0
    labelF['stress']=0.0
    labelF['strain']=0.0
    labelF['slope']=0.0
    
    n=0
    for i in result.index:
        if i in labelF['Lot'].values:
            labelF['~stress'][n]=result.loc[i]['Predicted Stress']
            labelF['~strain'][n]=result.loc[i]['Predicted Strain']
            labelF['~slope'][n]=result.loc[i]['Predicted Slope']
            labelF['stress'][n]=result.loc[i]['Stress']
            labelF['strain'][n]=result.loc[i]['Strain']
            labelF['slope'][n]=result.loc[i]['Slope']
            n+=1
            
    n=0
    for i in labelF['Lot'].values:
        if (datasetB[datasetB['lots']==i].size)!=0 :    
            temp=datasetB[datasetB['lots']==i]
            temp['~stress']=labelF['~stress'][n]
            temp['~strain']=labelF['~strain'][n]
            temp['~slope']=labelF['~slope'][n]
            temp['Porosity']=label['Porosity'][n]
            temp['Dispersity']=label['Dispersity'][n]
            temp['Size']=label['Size'][n]
            temp['Faceted']=label['Faceted'][n]
            temp['stress']=labelF['stress'][n]
            temp['strain']=labelF['strain'][n]
            temp['slope']=labelF['slope'][n]
            boxF=boxF.append(temp)
            del temp
            temp=pd.DataFrame()
            n+=1
            
    for i in lis:
        temp0=fuseX1(labelF, i)
        resultF=resultF.append(temp0)
        del temp0
        temp0=pd.DataFrame()
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
    plt.ylabel('%Error-slope')
    plt.title('Prediction error: HAF+SS')
    plt.legend('L')
    plt.grid()
    plt.show()
    
if mode==4:
    for i in lis:
        temp0=fuse2(label, i)
        result=result.append(temp0)
        del temp0
        temp0=pd.DataFrame()
    
    labelF=pd.read_csv('human.csv')
    labelF['~stress']=0.0
    labelF['~strain']=0.0
    labelF['~slope']=0.0
    labelF['stress']=0.0
    labelF['strain']=0.0
    labelF['slope']=0.0
    
    n=0
    for i in result.index:
        if i in labelF['Lot'].values:
            labelF['~stress'][n]=result.loc[i]['Predicted Stress']
            labelF['~strain'][n]=result.loc[i]['Predicted Strain']
            labelF['~slope'][n]=result.loc[i]['Predicted Slope']
            labelF['stress'][n]=result.loc[i]['Stress']
            labelF['strain'][n]=result.loc[i]['Strain']
            labelF['slope'][n]=result.loc[i]['Slope']
            n+=1
            
    n=0
    for i in labelF['Lot'].values:
        if (datasetB[datasetB['lots']==i].size)!=0 :    
            temp=datasetB[datasetB['lots']==i]
            temp['~stress']=labelF['~stress'][n]
            temp['~strain']=labelF['~strain'][n]
            temp['~slope']=labelF['~slope'][n]
            temp['Porosity']=label['Porosity'][n]
            temp['Dispersity']=label['Dispersity'][n]
            temp['Size']=label['Size'][n]
            temp['Faceted']=label['Faceted'][n]
            temp['stress']=labelF['stress'][n]
            temp['strain']=labelF['strain'][n]
            temp['slope']=labelF['slope'][n]
            boxF=boxF.append(temp)
            del temp
            temp=pd.DataFrame()
            n+=1
            
    for i in lis:
        temp0=fuseX2(labelF, i)
        resultF=resultF.append(temp0)
        del temp0
        temp0=pd.DataFrame()
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
    plt.ylabel('%Error-slope')
    plt.title('Prediction error: HAF+SS')
    plt.legend('L')
    plt.grid()
    plt.show()
    
if mode==5:
    for i in lis:
        temp0=haf2(label, i)
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
    plt.title('Prediction error: HAF')
    plt.legend('L')
    plt.grid()
    plt.show()
    
#resultF.to_csv('/Users/jaman1/Desktop/bsif_pdsf_sss.csv',index=False)