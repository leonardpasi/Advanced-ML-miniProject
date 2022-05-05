"""
Created on Sat Apr 30 16:36:53 2022

"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import signal

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


#%% Load data, extract features (compute PSD), prepare X and y

data = np.load('selected_channels.npy')
data = data[:,:,:,:8,:] # Keep only 8 flicker frequencies: [8, 9, 10, 11, 12, 13, 14, 15] Hz 
f_target = [8, 9, 10, 11, 12, 13, 14, 15]

# Another possibility: average channels in time domain. But performance is worse.
#data_chosen = np.mean(data_chosen, axis=(1), keepdims=True) 

# 35 subjects, 9 channels, 1500 samples, 8 flicker frequencies, 6 iterations
nb_subjects, nb_channels, nb_samples, nb_classes, nb_iterations = data.shape

fs = 250 # sampling frequency [Hz]

f, features = signal.periodogram(data, fs, axis=2)
features = np.mean(features, axis = 1) # Average channels in frequency domain


training_data = []
for subject in range(nb_subjects):
    for flicker in range(nb_classes):
        for itr in range(nb_iterations):
            training_data.append([features[subject,:,flicker,itr],flicker+1])
                
training_data = np.array(training_data,dtype=object)

X = np.array([i[0] for i in training_data]) 
y = np.array([i[1] for i in training_data])

f_low, f_high = 7.5, 31 # frequency range of interest
X = X[:,(f>=f_low) & (f<=f_high)] # reduce dimentionality of samples: from 751 to 142

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#%% Baseline: SVM classifier 82% accuracy

clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))


#%% PCA: further reducing the dimentionality does not seem necessary. But nice to see

pca = PCA(n_components=100)
pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_*100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance (%)') # n_components = 100 explains for >90% of the variance

#X_red = pca.transform(X_train)


#%% Gaussian Process Classifier

kernel = 1.0*kernels.RBF(5.0) # 86% accuracy. Amazing improvement wrt default kernel
clf = GaussianProcessClassifier(kernel)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))


#%% Grid-search with Adaboost

model = AdaBoostClassifier()

# define the grid of values to search
grid = dict()
grid['n_estimators'] = [300]
grid['learning_rate'] = [0.1]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1, cv=cv, scoring='accuracy', verbose=3)

# execute the grid search
grid_result = grid_search.fit(X, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
  
#%% Plot

def visualize_random(nb_samples=1, labels=[1,2]):
    """
    Plot randomly selected samples from desired classes
    
    Parameters
    ----------
    nb_samples : int, optional
        Number of randomly selected samples to plot, for each class.
    labels : list of int, optional
        List of classes to visualize. No more than 5 classes.

    Returns
    -------
    None.
    """
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for i,c in zip(labels, colors):
        X_i = X[y==i]
        random_indices = np.random.choice(X_i.shape[0], size=nb_samples, replace=False)
        ax.axvline(x=f_target[i-1], color=c, linewidth=1, linestyle='dashed', label="{}Hz".format(f_target[i-1]))
        ax.axvline(x=f_target[i-1]*2, color=c, linewidth=1, linestyle='dotted', label="{}Hz".format(2*f_target[i-1]))
        for j in random_indices:
            ax.plot(f[(f>=f_low) & (f<=f_high)], X_i[j], color=c, alpha=0.5, label="class {}".format(i))
            
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD')
    ax.legend()
    
visualize_random() # Useful observation : beyond 50 Hz, there's nothing to see. Why not reduce the dimentionality?


    
