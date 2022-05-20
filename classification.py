"""
Created on Sat Apr 30 16:36:53 2022

"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import signal
import joblib

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split


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

#%% Dimensionality reduction + quick test with SVM

ultra_reduce = True
f_low, f_high = 7.5, 31 # frequency range of interest
eps = 1e-6

if ultra_reduce:
    mask = np.zeros(f.shape, dtype=bool)
    for f_i in np.concatenate((np.asarray(f_target), np.asarray(f_target)*2)):
        idx = (np.abs(f - f_i)).argmin()
        mask[idx]=True
        
        # The following is meant for the 40 classes case, where many target flicker
        # frequencies are not exactly represented in the PSD. In that case we keep
        # the two closest frequencies. Another option is to change the resolution
        # of the periodogram, such that flicker frequencies are exactly represented
        # e.g. set "nfft = 1250" in the periodogram computation
        if f[idx] < f_i-eps:
            mask[idx+1]=True
        elif f[idx] > f_i+eps:
            mask[idx-1]=True
        
    X_red = X[:,mask] # dimensionality reduction: from 751 to 16 !
        
else:
    X_red = X[:,(f>=f_low) & (f<=f_high)] # dimensionality reduction: from 751 to 142

# Baseline: SVM

X_train, X_test, y_train, y_test = train_test_split(X_red, y, test_size=0.33, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

#%% PCA: classic approach to reduce dimensionality. Doesn't work at all here (wtf?)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pca = PCA(n_components=10)
pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_*100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance (%)')


clf = svm.SVC()
clf.fit(pca.transform(X_train), y_train)

y_pred = clf.predict(pca.transform(X_test))
print(accuracy_score(y_test,y_pred))


#%% Gaussian Process Classifier: 88.8% accuracy on dataset reduced to 16 dimensions

kernel = 1.0*kernels.RBF(5.0) #  Amazing improvement wrt default kernel
clf = GaussianProcessClassifier(kernel)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

#%% Quick tests to have a feeling hyperparameters ranges to explore in Grid Search

model = AdaBoostClassifier(n_estimators = 10000, learning_rate=0.001)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred))

#%% Grid-search with Adaboost. WARNING: this can take a while. It was executed once, and results were saved


model = AdaBoostClassifier()

# define the grid of values to search
grid = dict()
grid['n_estimators'] = [50, 100, 200, 400, 800, 1600, 3200, 6400]
grid['learning_rate'] = [1.0, 0.1, 0.01, 0.001]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1, cv=cv, 
                           scoring=['accuracy', 'f1_macro'], verbose=3, refit='accuracy',
                           return_train_score=True)

# execute the grid search
grid_result = grid_search.fit(X_red, y)

#%% Report Grid Search results

# Load results
grid_result = joblib.load('./Grid_Search_Decision_Stumps/grid_result_decision_stump.pkl')
grid = joblib.load('./Grid_Search_Decision_Stumps/grid_decision_stump.pkl')

test_acc_means = grid_result.cv_results_['mean_test_accuracy']
test_acc_stds = grid_result.cv_results_['std_test_accuracy']
train_acc_means = grid_result.cv_results_['mean_train_accuracy']
train_acc_stds = grid_result.cv_results_['std_train_accuracy']

test_f1_means = grid_result.cv_results_['mean_test_f1_macro']
test_f1_stds = grid_result.cv_results_['std_test_f1_macro']
train_f1_means = grid_result.cv_results_['mean_train_f1_macro']
train_f1_stds = grid_result.cv_results_['std_train_f1_macro']

test_time_means = grid_result.cv_results_['mean_score_time']
test_time_stds = grid_result.cv_results_['std_score_time']
train_time_means = grid_result.cv_results_['mean_fit_time']
train_time_stds = grid_result.cv_results_['std_fit_time']

params = grid_result.cv_results_['params']


print("\n \nBest accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Best F1 score: %f using %s" % (test_f1_means.max(), params[test_f1_means.argmax()]))

for i in range(len(params)):
    
    print("\nWith %r" % params[i])
    print("Accuracy -- Test : %f (%f) -- Train : %f (%f)" % (test_acc_means[i],
                                                             test_acc_stds[i],
                                                             train_acc_means[i],
                                                             train_acc_stds[i]))
    print("F1-score -- Test : %f (%f) -- Train : %f (%f)" % (test_f1_means[i],
                                                             test_f1_stds[i],
                                                             train_f1_means[i],
                                                             train_f1_stds[i]))
#%% HEAT MAPS

n1 = len(grid['learning_rate'])
n2 = len(grid['n_estimators'])
fig, (ax1, ax2) = plt.subplots(2,1)
fig.set_size_inches(8, 8)
fig.suptitle("Grid Search: Real Adaboost with decision stumps", size='x-large')

# TEST ACCURACY

acc = (test_acc_means*100).reshape((n1,n2)).round(decimals=1).astype(str)
std = (test_acc_stds*100).reshape((n1,n2)).round(decimals=1).astype(str)
annot = np.char.add(acc, "\n$\pm$")
annot = np.char.add(annot, std)
annot = annot.astype(object)

sns.heatmap(test_acc_means.reshape((n1,n2))*100, annot=annot, xticklabels=grid['n_estimators'],
            yticklabels=grid['learning_rate'], fmt='', ax=ax1, vmin=35, vmax=93, cbar=False)
ax1.set_title('Test accuracy: mean $\pm$ standard deviation')
ax1.set_ylabel('Learning rate')


# TRAIN ACCURACY

acc = (train_acc_means*100).reshape((n1,n2)).round(decimals=1).astype(str)
std = (train_acc_stds*100).reshape((n1,n2)).round(decimals=1).astype(str)
annot = np.char.add(acc, "\n$\pm$")
annot = np.char.add(annot, std)
annot = annot.astype(object)

sns.heatmap(train_acc_means.reshape((n1,n2))*100, annot=annot, xticklabels=grid['n_estimators'],
            yticklabels=grid['learning_rate'], fmt='', ax=ax2, vmin=35, vmax=93, cbar=False)
ax2.set_title('Train accuracy: mean $\pm$ standard deviation')
ax2.set_xlabel('Number of estimators')
ax2.set_ylabel('Learning rate')

# plt.savefig("Heat_Map_decision_stump.svg")

#%% Time complexity

fig, (ax1, ax2) = plt.subplots(2,1)
fig.set_size_inches(7, 8)
fig.suptitle("Real Adaboost: time complexity as a function of hyperparameters", size='x-large')

ax1.set_title('Training')
ax2.set_title('Testing')
ax2.set_xlabel('Number of estimators')
ax1.set_ylabel('Time [s]')
ax2.set_ylabel('Time [s]')
#ax1.set_xscale('log')

for i in range(n1):
    ax1.errorbar(grid['n_estimators'], train_time_means.reshape((n1,n2))[i,:], 
             label=grid['learning_rate'][i], yerr=train_time_stds.reshape((n1,n2))[i,:],
             marker='o', markersize=4)
    
    ax2.errorbar(grid['n_estimators'], test_time_means.reshape((n1,n2))[i,:], 
             label=grid['learning_rate'][i], yerr=test_time_stds.reshape((n1,n2))[i,:],
             marker='o', markersize=4)
    
ax1.legend(title='Learning rate')
ax2.legend(title='Learning rate')

#plt.savefig("ada_stumps_time_complexity_hyp.svg")

#%% Confusion Matrix for the selected model

model = AdaBoostClassifier(n_estimators = 200, learning_rate=0.1)
conf_matrices = []
kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

for train_index, test_index in kf.split(X_red, y):

   X_train, X_test = X_red[train_index], X_red[test_index]
   y_train, y_test = y[train_index], y[test_index]

   model.fit(X_train, y_train)
   conf_matrix = confusion_matrix(y_test, model.predict(X_test), normalize='true')
   conf_matrices.append(conf_matrix)
   
mean_conf_matrix = np.mean(conf_matrices, axis=0)
std_conf_matrix = np.std(conf_matrices, axis=0)

#%%
foo = mean_conf_matrix.round(decimals=2).astype(str)
bar = std_conf_matrix.round(decimals=2).astype(str)
annot = np.char.add(foo, "\n$\pm$")
annot = np.char.add(annot, bar)
annot = annot.astype(object)

fig, ax = plt.subplots() 
fig.set_size_inches(6, 6)  
sns.heatmap(mean_conf_matrix, annot=annot, fmt='', cbar=False, ax=ax)
        
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')
fig.suptitle('Confusion Matrix for Adaboost Classifier with decision \nstumps \
(n_estimators = 200, learning_rate=0.1)')

#plt.savefig("Conf_Matrix_AdaBoost_stump.svg")                                

#%% Plot

def visualize_random(nb_samples=1, labels=[1,3]):
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
        X_i = X_i[:, (f>=f_low) & (f<=f_high)]
        random_indices = np.random.choice(X_i.shape[0], size=nb_samples, replace=False)
        ax.axvline(x=f_target[i-1], color=c, linewidth=1, linestyle='dashed', label="{}Hz".format(f_target[i-1]))
        ax.axvline(x=f_target[i-1]*2, color=c, linewidth=1, linestyle='dotted', label="{}Hz".format(2*f_target[i-1]))
        for j in random_indices:
            ax.plot(f[(f>=f_low) & (f<=f_high)], X_i[j], color=c, alpha=0.5, label="class {}".format(i))
            
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD')
    ax.legend()
    
def visualize_mean(save_fig=False):
    """
    Cool function to visualize mean PSD for each flicker frequency :)
    """
    
    fig, axs = plt.subplots(4,2)
    fig.set_size_inches(10, 10)
    fig.suptitle("Mean Power Spectral Density for each flicker frequency")
    fig.text(0.5, 0.06, 'Hz', ha='center', va='center')
    fig.text(0.06, 0.5, 'Power Spectral Density', ha='center', va='center', rotation='vertical')
    
    for i, ax in zip(np.unique(y), axs.flatten(order='F')):
        X_i = X[y==i]
        psd_mean = X_i[:, (f>=f_low) & (f<=f_high)].mean(axis=0);
        psd_std = X_i[:, (f>=f_low) & (f<=f_high)].std(axis=0)
        ax.plot(f[(f>=f_low) & (f<=f_high)], psd_mean)
        ax.fill_between(f[(f>=f_low) & (f<=f_high)], psd_mean-psd_std, psd_mean+psd_std, alpha=0.3, label='+/- $\sigma$')
        ax.axvline(x=f_target[i-1],linewidth=1, linestyle='dashed', label="{}Hz".format(f_target[i-1]))
        ax.axvline(x=f_target[i-1]*2, linewidth=1, linestyle='dotted', label="{}Hz".format(2*f_target[i-1]))
        ax.set_ylim([-2,16])
        ax.legend()
    
    if save_fig :
        plt.savefig("Mean_PSDs.svg")
    
#visualize_random()
visualize_mean()


    
