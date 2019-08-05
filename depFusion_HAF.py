#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:09:16 2019

@author: jaman1
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
import warnings
warnings.filterwarnings("ignore")


def fuse1(data, leaveout_lot):

        model0=LinearRegression()
        model1=LinearRegression()
        model2=LinearRegression()
        sys.stdout.flush()
        training = data.values[data['lots']!=leaveout_lot]
        testing = data.values[data['lots']==leaveout_lot]

        X=np.vstack(training[:,1:11])
        input_train_data = X
        output_train_data0 = training[:,11]
        output_train_data1 = training[:,12]
        output_train_data2 = training[:,13]
        X_=np.vstack(testing[:,1:11])
        input_test_data=X_
        
        model0.fit(input_train_data, output_train_data0)
        model1.fit(input_train_data, output_train_data1)
        model2.fit(input_train_data, output_train_data2)
        
        ##################################################################
        predicted_stress=0
        predicted_stress = model0.predict(input_test_data)
        error0=abs(testing[0,11]-predicted_stress)
        pError0=100*error0/testing[0,11]
        
        ##################################################################
        predicted_strain=0
        predicted_strain = model1.predict(input_test_data)
        error1=abs(testing[0,12]-predicted_strain)
        pError1=100*error1/testing[0,12]
        
        ##################################################################
        predicted_slope=0
        predicted_slope = model2.predict(input_test_data)
        error2=abs(testing[0,13]-predicted_slope)
        pError2=100*error2/testing[0,13]
        ##################################################################
        
        # Round to 2 decimal place        
        temp = {'Stress':{leaveout_lot:np.around(testing[0,11],2)},
                'Predicted Stress':{leaveout_lot:np.around(predicted_stress,2)},
                'Absolute error0':{leaveout_lot:np.around(error0,2)}, 
                '%Error0':{leaveout_lot:np.around(pError0,2)},
                'Strain':{leaveout_lot:np.around(testing[0,12],2)},
                'Predicted Strain':{leaveout_lot:np.around(predicted_strain,2)},
                'Absolute error1':{leaveout_lot:np.around(error1,2)}, 
                '%Error1':{leaveout_lot:np.around(pError1,2)},
                'Slope':{leaveout_lot:np.around(testing[0,13],2)},
                'Predicted Slope':{leaveout_lot:np.around(predicted_slope,2)},
                'Absolute error2':{leaveout_lot:np.around(error2,2)},
                '%Error2':{leaveout_lot:np.around(pError2,2)}} 
        
        info=pd.DataFrame.from_dict(temp)
        return info
    
def fuseX1X(data):
     
        model2=LinearRegression()

        sys.stdout.flush()
        training = data.values

        X=np.vstack(training[:,11:14])
        input_train_data = X
        output_train_data2 = training[:,16]
        model2.fit(input_train_data, output_train_data2)

        return model2