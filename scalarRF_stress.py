#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################
# Created on Thu Jun 13 12:32:07 2019
#
# @author: jaman1@llnl.gov
# Resource used: RandomForestSnippet.py (loveland4@llnl.gov)
###############################################################


import pickle
import matplotlib.pyplot as plt
from sklearn import ensemble
#import os
import sys
import pandas as pd 
import numpy as np
#import scipy

def RF(data, leaveout_lot):
#def RF(data, leaveout_lot, bsif_flag=False):    
        """
        Run a RF regression model on the data

        Arguments:
        data: The dataframe you will be using for your regression (pandas)
        leaveout_lot: The lot you will be leaving out (string)
       
        """
        #print(" - Leaving out {0} for testing.".format(leaveout_lot))
        sys.stdout.flush()
        ##################################################
        training = data.values[data.index!=leaveout_lot]
        testing = data.values[data.index==leaveout_lot]
        ##################################################
        #print(training.shape)
        #print(testing.shape)
        ##################################################
        verbose=0
        model = ensemble.RandomForestRegressor(n_estimators=100, max_depth=32, verbose=verbose, max_features=0.33, random_state=99, n_jobs=-1)

        # We use this for setting up the input training data with just VLAD features
        """
        if (bsif_flag == False):
            input_train_data = np.zeros((len(training['vlad']), len(np.array(training['vlad'])[0])))
            for counter, data in enumerate(np.array(training['vlad'])):
                input_train_data[counter, :] = data
        
        # We use this for setting up input data with VLAD and BSIF features
        if (bsif_flag == True):
            input_train_data = np.zeros((len(training['vlad']), len(np.array(training['vlad'])[0]) + len(np.array(training['bsif'])[0])))
            for counter in range(len(np.array(training['vlad']))):
                input_train_data[counter, :] = np.append(np.array(training['vlad'])[counter], np.array(training['bsif'])[counter])     
        print(input_train_data.shape)
        """
        ##################################
        input_train_data = training[:,0:2]
        #print(input_train_data.shape)
        
        # We use this for setting up the output training data
        #output_train_data = (np.array(training['lot_label']))
        output_train_data = training[:,2]
        #print(output_train_data.shape) 
        ##################################
        
        """"
        # We use this for setting up the input test data 
        if (bsif_flag == False):
            input_test_data = np.zeros((len(testing['vlad']), len(np.array(testing['vlad'])[0])))
            for counter, data in enumerate(np.array(testing['vlad'])):
                input_test_data[counter, :] = data
        
        if (bsif_flag == True):
            input_test_data = np.zeros((len(testing['vlad']), len(np.array(testing['vlad'])[0]) + len(np.array(testing['bsif'])[0])))
            for counter in range(len(np.array(testing['vlad']))):
                input_test_data[counter, :] = np.append(np.array(testing['vlad'])[counter], np.array(testing['bsif'])[counter])
        """
        ###################################
        input_test_data=testing[:,0:2]
        #output_test_data=testing[:,2]
        #print(input_test_data.shape)

        #print(' - Fitting RF model')
        model.fit(input_train_data, output_train_data)
        """
        try:
            os.mkdir('./Models')
        except:
            print('Models folder already created')
        model_file_path = "./Models/model_{0}.pkl".format(self.unique_folder)
        pickle.dump(model, open(model_file_path,'wb'))
        """
        ###################################
        predicted_stress = model.predict(input_test_data)
        error=abs(testing[:,2]-predicted_stress)
        pError=100*error/testing[:,2]
        # Round to 2 decimal place
        temp = {'Slope':{leaveout_lot:testing[:,0].round(2)}, 'Strain':{leaveout_lot:testing[:,1].round(2)}, 'Stress':{leaveout_lot:testing[:,2].round(2)}, 'Predicted Stress':{leaveout_lot:predicted_stress.round(2)}, 'Absolute error':{leaveout_lot:error.round(2)}, '%Error':{leaveout_lot:pError.round(2)}} 
        info=pd.DataFrame.from_dict(temp)
        return info
#def combine_df(VLAD_df, BSIF_df):
#        combined_df = pd.merge(VLAD_df, BSIF_df, left_on=['lot','stub','file'], right_on=['lot', 'stub', 'file'])
#        return combined_df
        
    
#SEM_output_data.pkl should be in working directory
pickle_in=open("SEM_output_data.pkl","rb")
sem_data=pickle.load(pickle_in)
print("\nPrinting raw dictionary: \n")
print(sem_data)

dataset=pd.DataFrame.from_dict(sem_data)
print("\nPrinting dataframe extracted from dictionary: \n")
print(dataset)
print("\nPrinting generic statistics: \n")
print(dataset.describe())
print("\n\n")

result=RF(dataset, dataset.index[0])
for i in dataset.index:
    temp=RF(dataset, i)
    result=result.append(temp)
result = result[1:dataset.index.size+1]
print(result)

MAPE=0
RMSE=0
MGT=0

# Measuring error index
for i in result.index:
    MAPE+=result.loc[i,'%Error']
    RMSE+=(result.loc[i,'Absolute error'])**2
    MGT+=result.loc[i,'Stress']
MAPE/=result.index.size
RMSE/=result.index.size
RMSE=RMSE**(0.5)
MGT/=result.index.size
NRMSE=RMSE/MGT
print("\n\n")
print("MAPE: {0}".format(np.around(MAPE,2)))
print("RMSE: {0}".format(np.around(RMSE,2)))
print("NRMSE: {0}".format(np.around(NRMSE,2)))
print("\n\n")

#Plot/Compare
sort=result.sort_values(by="%Error")
plt.plot(sort.loc[:,'%Error'],'r-o')
plt.xlabel('Material Lot')
plt.ylabel('%Error')
plt.title('Prediction error: RF')
plt.legend('R')
plt.grid()
plt.show()