"""
Created on Sat Apr 30 16:36:53 2022

"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import signal
from scipy.fft import fft

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


#%%

data_chosen = np.load('selected_channels.npy')

# Optional: mean across channels to reduce noise, in time domain
# data_chosen = np.mean(data_chosen, axis=(1), keepdims=True) 

nb_subjects, nb_channels, nb_samples, nb_classes, nb_iterations = data_chosen.shape

# 35 subjects, 9 channels, 1500 samples, 40 flicker frequencies, 6 iterations


#%%

# PREPROCESSING FUNCTIONS

def band_pass_filter(ntaps, low_f ,high_f, fs, window):
    # Return weights of digital band pass filter
    
    taps_hamming = signal.firwin(ntaps, [low_f, high_f], fs=fs, 
                                 pass_zero=False, window=window, scale=False)
    return taps_hamming


def Feature_extract(data, low_f=5, high_f=45, fs=250, filter_type="hamming", ntaps=128):
    """
    Description
    ----------
    Compute relevant features from the chosen data

    Returns
    -------
    F : array of complex (same dimension as input data)
        Fast Fourier Transform of the signals
    PSD : array of float
        Power Spectral Density of the signals (one sided)
    PSD_filter : array of float
        PSD of the signals after band pass filter
    f : array of float (1D)
        Sample frequencies

    """
    F = fft(data, axis=2)
    f, PSD = signal.periodogram(data, fs, axis=2)
    
    # Get frequency response of band pass filter
    taps_hamming = band_pass_filter(ntaps,low_f,high_f,fs,str(filter_type))
    _, h = signal.freqz(taps_hamming, 1, worN=PSD.shape[2])
    
    # Filtering
    PSD_filter = np.copy(PSD)
    for subject in range(nb_subjects):  
        for channel in range(nb_channels):
            for flicker in range(nb_classes):
                for itr in range(nb_iterations):
                    PSD_filter[subject,channel,:,flicker,itr] = PSD[subject,channel,:,flicker,itr]*np.square(np.abs(h))
                    
    return F, PSD, PSD_filter, f

F, PSD, features, f = Feature_extract(data_chosen)

#%% Creating training and testing data (frequency domain)

training_data = []

for subject in range(nb_subjects):
    for flicker in range(nb_classes):
        for itr in range(nb_iterations):
            training_data.append([features[subject,:,:,flicker,itr],flicker+1])
                
training_data = np.array(training_data,dtype=object)

X = np.array([i[0] for i in training_data]) 
y = np.array([i[1] for i in training_data])

Xavg = np.mean(X, axis=1) # Average across channels

X_train, X_test, y_train, y_test = train_test_split(Xavg,y,train_size=0.7,random_state=123)

#del training_data, X, y, Xavg

#%% Baseline: SVM classifier 70% accuracy

clf = svm.SVC()
clf.fit(X_train, y_train)

#%% Adaboost with decision tree

clf = AdaBoostClassifier(n_estimators=500, learning_rate=1)
clf.fit(X_train, y_train)

#%% PCA (necessary before Gaussian Process Classifier)

pca = PCA(n_components=100)
pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_*100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance (%)') # n_components = 100 explains for >90% of the variance

X_red = pca.transform(X_train)


#%% Gaussian Process Classifier

clf = GaussianProcessClassifier()

#clf.fit(X_red, y_train) # MF keeps crushing on me
clf.fit(X_red, y_train>20) # Works fine with binary classification

#%% Test and get accuracy (following PCA and GPC)

y_pred = clf.predict(pca.transform(X_test))
print(accuracy_score(y_test>20,y_pred))

#%% Grid-search with Adaboost

model = AdaBoostClassifier()

# define the grid of values to search
grid = dict()
grid['n_estimators'] = [400, 500]
grid['learning_rate'] = [0.05, 0.1]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1, cv=cv, scoring='accuracy', verbose=3)

# execute the grid search
grid_result = grid_search.fit(Xavg, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
  
