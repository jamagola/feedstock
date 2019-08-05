#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:14:17 2019

@author: jaman1
"""
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Ridge
from sklearn import ensemble
import warnings
warnings.filterwarnings("ignore")

def fuse1(box, label, leaveout_lot, RF, BSIF, HAF, INST):

        #verbose=0
        #model2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        #Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
        #model2 = Ridge(alpha=.5)
    
        if RF==True:
            verbose=0
            model0 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
            model1 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
            model2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
            
        else:   
            model0=LinearRegression()
            model1=LinearRegression()
            model2=LinearRegression()
       
        
        sys.stdout.flush()
        
        if BSIF==True:
            data=box
            s1=4
            s2=5
            s3=6
        else:
            data=label
            s1=13
            s2=14
            s3=15
        
        training = data.values[data['lots']!=leaveout_lot]
        testing = data.values[data['lots']==leaveout_lot]
        
        
        if BSIF and HAF and INST:
            
            if RF==True:
                cnt=0
                for i in training[:,0]:
                    training[:,0][cnt]=np.append(i[0:20], training[:,7:19][cnt])
                    cnt+=1
                
                cnt=0
                for i in testing[:,0]:
                   testing[:,0][cnt]=np.append(i[0:20], testing[:,7:19][cnt])
                   cnt+=1
            
            else:
                cnt=0
                for i in training[:,0]:
                    training[:,0][cnt]=np.append(i, training[:,7:19][cnt])
                    cnt+=1
                
                cnt=0
                for i in testing[:,0]:
                   testing[:,0][cnt]=np.append(i, testing[:,7:19][cnt])
                   cnt+=1
            
            X=np.vstack(training[:,0])
            X_=np.vstack(testing[:,0])
            

        if BSIF and HAF and (not INST):
            if RF==True:
                cnt=0
                for i in training[:,0]:
                    training[:,0][cnt]=np.append(i[0:20], training[:,7:12][cnt])
                    cnt+=1
                
                cnt=0
                for i in testing[:,0]:
                   testing[:,0][cnt]=np.append(i[0:20], testing[:,7:12][cnt])
                   cnt+=1
            
            else:
                cnt=0
                for i in training[:,0]:
                    training[:,0][cnt]=np.append(i, training[:,7:12][cnt])
                    cnt+=1
                
                cnt=0
                for i in testing[:,0]:
                   testing[:,0][cnt]=np.append(i, testing[:,7:12][cnt])
                   cnt+=1
            
            X=np.vstack(training[:,0])
            X_=np.vstack(testing[:,0])
            
        if BSIF and INST and (not HAF):
            if RF==True:
                cnt=0
                for i in training[:,0]:
                    training[:,0][cnt]=np.append(i[0:20], training[:,12:19][cnt])
                    cnt+=1
                
                cnt=0
                for i in testing[:,0]:
                   testing[:,0][cnt]=np.append(i[0:20], testing[:,12:19][cnt])
                   cnt+=1
            
            else:
                cnt=0
                for i in training[:,0]:
                    training[:,0][cnt]=np.append(i, training[:,12:19][cnt])
                    cnt+=1
                
                cnt=0
                for i in testing[:,0]:
                   testing[:,0][cnt]=np.append(i, testing[:,12:19][cnt])
                   cnt+=1
                   
            X=np.vstack(training[:,0])
            X_=np.vstack(testing[:,0])
        
        if BSIF and (not INST) and (not HAF):
            if RF==True:
                cnt=0
                for i in training[:,0]:
                    training[:,0][cnt]=i[0:20]
                    cnt+=1
                
                cnt=0
                for i in testing[:,0]:
                   testing[:,0][cnt]=i[0:20]
                   cnt+=1
            
            X=np.vstack(training[:,0])
            X_=np.vstack(testing[:,0])
            
        if HAF and INST and (not BSIF):          
            X=np.vstack(training[:,1:13])
            X_=np.vstack(testing[:,1:13])
            
        if HAF and (not INST) and (not BSIF):          
            X=np.vstack(training[:,8:13])
            X_=np.vstack(testing[:,8:13])
            
        if INST and (not HAF) and (not BSIF):          
            X=np.vstack(training[:,1:8])
            X_=np.vstack(testing[:,1:8])

        input_train_data = X
        output_train_data0 = training[:,s1]
        output_train_data1 = training[:,s2]
        output_train_data2 = training[:,s3]
        input_test_data=X_
        
        model0.fit(input_train_data, output_train_data0)
        model1.fit(input_train_data, output_train_data1)
        model2.fit(input_train_data, output_train_data2)
        
        ##################################################################
        predicted_stress=0
        stress=0;
        count=0;
        if BSIF==True:
            for i in input_test_data:
                stress = model0.predict(np.array([input_test_data[count]]))
                predicted_stress=np.append(predicted_stress,stress)
                count+=1
            predicted_stress=np.median(predicted_stress)
        else:
            predicted_stress = model0.predict(input_test_data)
        error0=abs(testing[0,s1]-predicted_stress)
        pError0=100*error0/testing[0,s1]
        
        ##################################################################
        predicted_strain=0
        strain=0;
        count=0;
        if BSIF==True:
            for i in input_test_data:
                strain = model1.predict(np.array([input_test_data[count]]))
                predicted_strain=np.append(predicted_strain,strain)
                count+=1
            predicted_strain=np.median(predicted_strain)
        else:
            predicted_strain = model1.predict(input_test_data)
        error1=abs(testing[0,s2]-predicted_strain)
        pError1=100*error1/testing[0,s2]
        
        ##################################################################
        predicted_slope=0
        slope=0;
        count=0;
        if BSIF==True:
            for i in input_test_data:
                slope = model2.predict(np.array([input_test_data[count]]))
                predicted_slope=np.append(predicted_slope,slope)
                count+=1
            predicted_slope=np.median(predicted_slope)
        else:
            predicted_slope = model2.predict(input_test_data)
        error2=abs(testing[0,s3]-predicted_slope)
        pError2=100*error2/testing[0,s3]
        ##################################################################
        
        # Round to 2 decimal place        
        temp = {'Stress':{leaveout_lot:np.around(testing[0,s1],2)},
                'Predicted Stress':{leaveout_lot:np.around(predicted_stress,2)},
                'Absolute error0':{leaveout_lot:np.around(error0,2)}, 
                '%Error0':{leaveout_lot:np.around(pError0,2)},
                'Strain':{leaveout_lot:np.around(testing[0,s2],2)},
                'Predicted Strain':{leaveout_lot:np.around(predicted_strain,2)},
                'Absolute error1':{leaveout_lot:np.around(error1,2)}, 
                '%Error1':{leaveout_lot:np.around(pError1,2)},
                'Slope':{leaveout_lot:np.around(testing[0,s3],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        
        info=pd.DataFrame.from_dict(temp)
        return info
