#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 08:46:00 2022

@author: jonasvishart
"""

#%% Computational plots
#https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Font settings
font = {'family' : 'Georgia',
       'weight'  : 'normal',
       'size'    :  22}
plt.rc('font',**font)


#plot with error bars and standard deviation

def computational_timer(model, n_samples, n_features, n_itr):
    #check if we are running over samples or dimension
    if len(n_samples)>1:
        samples = n_samples
    else :
        samples = n_features
    
    #Data sizes
    exctrain_time = np.zeros((n_itr,len(samples)))
    exctest_time = np.zeros((n_itr,len(samples)))

        
    for itr in range(n_itr):
        for n in range(len(samples)):  
            # define dataset
            #check if we are running over samples or dimension
            if len(n_samples)>1:
                X, y = make_classification(n_samples=n_samples[n], n_features=n_features[0], n_informative=15, n_redundant=5, random_state=1)
            else:
                X, y = make_classification(n_samples=n_samples[0], n_features=n_features[n], n_informative=15, n_redundant=5, random_state=1)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
            # recor training time
            start_train = time.time()
            # fit the model
            model.fit(X_train, y_train)
            # record current time
            end_train = time.time()
            
            # record testing time
            start_test = time.time()
            model.predict(X_test)
            end_test = time.time()
            # report execution time
            result_train = end_train - start_train #[s]
            result_test = end_test - start_test #[s]
            
            
            exctrain_time[itr,n] = result_train
            exctest_time[itr,n] = result_test
    
    avg_exctrain_time=np.mean(exctrain_time,axis=0)
    avg_exctest_time=np.mean(exctest_time,axis=0)
    
    std_exctrain_time=np.std(exctrain_time,axis=0)
    std_exctest_time=np.std(exctest_time,axis=0)
            
    return avg_exctrain_time, avg_exctest_time, std_exctrain_time, std_exctest_time

# define model
model_adaboost = AdaBoostClassifier()
model_gauss = GaussianProcessClassifier(1*kernels.RBF(4.0) )
model_svm = svm.SVC()

#%% Computational cost - Sample size 
# compute computational time
n_samples = [10, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000]
n_features = [20]
n_itr = 5

traintime_adaboost, testtime_adaboost, std_traintime_adaboost, std_testtime_adaboost = computational_timer(model_adaboost,n_samples,n_features,n_itr)
traintime_gauss, testtime_gauss, std_traintime_gauss, std_testtime_gauss  = computational_timer(model_gauss,n_samples,n_features,n_itr)
traintime_svm, testtime_svm, std_traintime_svm, std_testtime_svm = computational_timer(model_svm,n_samples,n_features,n_itr)
#%%
model_adaboost_2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5))
traintime_adaboost_2, testtime_adaboost_2, std_traintime_adaboost_2, std_testtime_adaboost_2 = computational_timer(model_adaboost_2,n_samples,n_features,n_itr)

#%%

# plots
# training time

plt.errorbar(np.array(n_samples)/2, traintime_svm, yerr = std_traintime_svm, 
             uplims = True, 
             lolims = True,
             label ='SVM')

plt.errorbar(np.array(n_samples)/2, traintime_adaboost, yerr = std_traintime_adaboost, 
             uplims = True, 
             lolims = True,
             label ='AdaBoost V1)')

plt.errorbar(np.array(n_samples)/2, traintime_adaboost_2, yerr = std_traintime_adaboost_2, 
             uplims = True, 
             lolims = True,
             label ='AdaBoost V2')

plt.errorbar(np.array(n_samples)/2, traintime_gauss, yerr = std_traintime_gauss, 
             uplims = True, 
             lolims = True,
             label ='GPC')


plt.legend()
plt.xlabel("No. of samples")
plt.ylabel("Train time [s]")
plt.savefig("TrainSamples_2.png",bbox_inches="tight")
plt.show()

# test time
plt.errorbar(np.array(n_samples)/2, testtime_svm, yerr = std_testtime_svm, 
             uplims = True, 
             lolims = True,
             label ='SVM')

plt.errorbar(np.array(n_samples)/2, testtime_adaboost, yerr = std_testtime_adaboost, 
             uplims = True, 
             lolims = True,
             label ='AdaBoost V1')

plt.errorbar(np.array(n_samples)/2, testtime_adaboost_2, yerr = std_testtime_adaboost_2, 
             uplims = True, 
             lolims = True,
             label ='AdaBoost V2')

plt.errorbar(np.array(n_samples)/2, testtime_gauss, yerr = std_testtime_gauss, 
             uplims = True, 
             lolims = True,
             label ='GPC')



plt.legend()
plt.xlabel("No. of samples")
plt.ylabel("Test time [s]")
plt.savefig("TestSamples.png",bbox_inches="tight")
plt.show()

#%% Computational cost - Dimension
# compute computational time
n_samples = [200]
n_features = [20, 500, 750, 1000, 1250, 1500, 1750, 2000,2250, 2500]
n_itr = 5

traintime_adaboost, testtime_adaboost, std_traintime_adaboost, std_testtime_adaboost = computational_timer(model_adaboost,n_samples,n_features,n_itr)
traintime_gauss, testtime_gauss, std_traintime_gauss, std_testtime_gauss  = computational_timer(model_gauss,n_samples,n_features,n_itr)
traintime_svm, testtime_svm, std_traintime_svm, std_testtime_svm = computational_timer(model_svm,n_samples,n_features,n_itr)
#%%  

# plots
# training time
plt.errorbar(n_features, traintime_adaboost, yerr = std_traintime_adaboost, 
             uplims = True, 
             lolims = True,
             label ='AdaBoost')

plt.errorbar(n_features, traintime_gauss, yerr = std_traintime_gauss, 
             uplims = True, 
             lolims = True,
             label ='GPC')
plt.errorbar(n_features, traintime_svm, yerr = std_traintime_svm, 
             uplims = True, 
             lolims = True,
             label ='SVM')
plt.legend()
plt.xlabel("No. of features")
plt.ylabel("Train time [s]")
plt.savefig("TrainFeatures.pdf",bbox_inches="tight")
plt.show()

# test time
plt.errorbar(n_features, testtime_adaboost, yerr = std_testtime_adaboost, 
             uplims = True, 
             lolims = True,
             label ='AdaBoost')

plt.errorbar(n_features, testtime_gauss, yerr = std_testtime_gauss, 
             uplims = True, 
             lolims = True,
             label ='GPC')
plt.errorbar(n_features, testtime_svm, yerr = std_testtime_svm, 
             uplims = True, 
             lolims = True,
             label ='SVM')
plt.legend()
plt.xlabel("No. of features")
plt.ylabel("Test time [s]")
plt.savefig("TestFeatures.pdf",bbox_inches="tight")
plt.show()



