'''
Created on Jul 6, 2016

@author: hugomathien
'''
from __future__ import print_function

from time import time
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import ShuffleSplit
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.multiclass import OneVsRestClassifier
from sklearn import neighbors, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


learningVectorFolder = "/Users/hugomathien/Documents/workspace/footballdata/learning_vector/"
learningVectorFile = "learn_2_class_53_features.txt"
use_pca = False
searchGrid = False


print(__doc__)

def plotGridSearchCV(clf):
    ###############################################################################
    # HEATMAP SVM PARAMETERS
    ###############################################################################
    # plot the scores of the grid
    # grid_scores_ contains parameter settings and scores
    # We extract just the scores
    scores = [x[1] for x in clf.grid_scores_]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    
    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.
    
    plt.figure(figsize=(6, 5))
    plt.subplots_adjust(left=.2, right=0.75, bottom=0.40, top=0.75)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.get_cmap('hot'),
               norm=MidpointNormalize(vmin=0.1, midpoint=0.60))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

###############################################################################
# Load data
X_raw, y = load_svmlight_file(learningVectorFolder + learningVectorFile)
#X = normalize(X_raw, norm='l2', axis=1, copy=True)
scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X_raw)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = sel.fit_transform(X)
n_classes = y[0]
n_samples, n_features = X.shape
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# PCA SPECTRUM
###############################################################################
if use_pca:
    pca = decomposition.PCA()
    X_dense = X.toarray()
    pca.fit(X_dense)
    plt.figure(1, figsize=(6, 5))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.show()
    print('Plotted PCA')
    
    pca.n_components = 40
    X = pca.fit_transform(X_dense)
###############################################################################
# CROSS VALIDATION SET
###############################################################################
cv = ShuffleSplit(n_samples, n_iter=2, test_size=0.4, random_state=50)
for train_index, test_index in cv:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

###############################################################################
# TRAIN A SVM CLASSIFIER ON THE CROSS VALIDATION SET
###############################################################################
print("Fitting the classifier to the training set")
t0 = time() 

# Run a search grid to find the optimum parameters
if searchGrid:
    #C = 10, gamma = 0.0001
    #C_range = np.logspace(-5, 5, 6)
    #gamma_range = np.logspace(-3, 3, 6)
    svc = SVC(kernel='rbf', decision_function_shape='ovr')
    C_range = [1e1, 30, 60, 1e2]
    gamma_range = [0.000001,0.00001,0.0001]
    param_grid = {'kernel': ['rbf'],
              'C': C_range,
              'gamma': gamma_range}
    clf = GridSearchCV(svc, param_grid,verbose=10,cv=cv)
    clf = clf.fit(X, y)
    
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
    plotGridSearchCV(clf)
    
# Run an outright SVC
else:
    svc = SVC(kernel='rbf', decision_function_shape='ovr',C=10,gamma=0.001)
    svc.fit(X_train,y_train)
    score = svc.score(X_test,y_test)
    print("TEST SCORE = " + str(score))
    print (svc.predict(X_test))
    decisionFunction = svc.decision_function(X_test)
    print(decisionFunction)
