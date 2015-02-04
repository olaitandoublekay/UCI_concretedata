
#code by Shiv Surya
#performs regression with l2-regularization
#import necessary packages

import csv
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer,mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from readData import readData
from gridsearch_helper import grid_search_helper_basic,grid_search_helper


#read the sets split into training,testing and validation sets by dataSplit.py

X_train,Y_train,X_val,Y_val,X_test,Y_test=readData()



#define the no of dimensions of the feature vector and the max dimension of the expanded polynomial
#feature space
noDim=8
noPoly=2

#define no of repetitions and folds of Cross-Validation that is to be done for training
noRep=5
n_folds=5



#define scoring functions
mse_scorer=make_scorer(mean_squared_error,greater_is_better=False,needs_proba=False,needs_threshold=False)
r2_scorer=make_scorer(r2_score,greater_is_better=True,needs_proba=False,needs_threshold=False)


#Define a parameter grid for exhaustive Grid Search. For ridge the parameter space has alpha as it's member

r2Poly=np.zeros((noPoly,1))

#perform N-Fold CV with "noRep" repetitions
for i in range(1,noPoly+1):
    #apply transform only if the polynomial is greater than 1
    if(i>1):
        pfeat = PolynomialFeatures(degree=i,include_bias=False)
        X = pfeat.fit_transform(X_train)
    else:
        X = X_train
    print X.shape[1]
    param_grid={'n_nonzero_coefs':range(1,X.shape[1])}
    grid_values= np.array(param_grid.values()).astype('int')
    print grid_values
    gridSize=grid_values.shape[1]
    OMP_regr = OrthogonalMatchingPursuit()
    cv_grid_scores_mean,cv_grid_scores_std = grid_search_helper(X, Y_train,OMP_regr, param_grid,gridSize, n_jobs=-1,
                                               n_folds=n_folds, n_repetitions=noRep, scoring=mse_scorer, iid=False)
    #Find optimum val and calculate r2_score on testing set or each polynomial
    opt_ind = np.where(-cv_grid_scores_mean == np.amin(-cv_grid_scores_mean))
    OMP_regr= OrthogonalMatchingPursuit(n_nonzero_coefs= grid_values[0,opt_ind])
    print X.shape
    OMP_regr.fit(X,Y_train)
    print OMP_regr.score(X,Y_train)

    if(i>1):

        X_val_tr = pfeat.fit_transform(X_val)
    else:
        X_val_tr = X_val

    r2Poly[i-1]=OMP_regr.score(X_val_tr,Y_val)

print r2Poly




