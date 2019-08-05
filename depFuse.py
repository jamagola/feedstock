#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:09:16 2019

@author: jaman1
"""

#import pickle
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
#from sklearn import ensemble
import warnings
warnings.filterwarnings("ignore")


def fuse1(data1, data2, leaveout_lot):

        model0=LinearRegression()
        model1=LinearRegression()
        model2=LinearRegression()
        #verbose=0
        #model2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        #Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
        #model2 = Ridge(alpha=.5)
        
        data=data1 #bsif
        
        sys.stdout.flush()
        training = data.values[data['lots']!=leaveout_lot]
        testing = data.values[data['lots']==leaveout_lot]
        
        """
        cnt=0
        for i in training[:,0]:
            training[:,0][cnt]=np.append(i, training[:,7:17][cnt])
            #training[:,0][cnt][0:20]=np.append(i, training[:,4:6][cnt])
            cnt+=1
        
        cnt=0
        for i in testing[:,0]:
           testing[:,0][cnt]=np.append(i, testing[:,7:17][cnt])
           cnt+=1
        """

        X=np.vstack(training[:,0])
        #X=np.vstack(training[:,0])[:,0:20]
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
    
def fuseX1X(data):
     
        #model0=LinearRegression()
        #model1=LinearRegression()
        model2=LinearRegression()
        #verbose=0
        #model2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)
        #Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
        #model2 = Ridge(alpha=.5)

        sys.stdout.flush()
        training = data.values
        
        
        cnt=0
        for i in training[:,0]:
            training[:,0][cnt]=np.append(i, training[:,17:20][cnt])
            #training[:,0][cnt][0:20]=np.append(i, training[:,4:6][cnt])
            cnt+=1
        
        X=np.vstack(training[:,0])
        #X=np.vstack(training[:,0])[:,0:20]
        input_train_data = X
        output_train_data2 = training[:,4] #stress
        model2.fit(input_train_data, output_train_data2)

        return model2