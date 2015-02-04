#code written by Shiv Surya
#wrappper around GridSearchCV for implementing CV with multiple repetitions

import numpy as np
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.grid_search import GridSearchCV


def grid_search_helper_basic(X,y,estimator,param_grid,n_folds=5,n_jobs=-1,pre_dispatch='2*noJobs',scoring=None,verbose=0,fit_params=None,
                         score_func=None,r_state=None):


    if (r_state == None):
        r_state = np.empty(1,dtype=object)

    cv=KFold(X.shape[0],n_folds,shuffle=True,random_state=r_state[0])

    gridsearch = GridSearchCV(estimator, param_grid, scoring=None, loss_func=None,
                 score_func=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs')
    gridsearch.fit(X, y)

    return gridsearch


def grid_search_helper( X, y,estimator, param_grid, gridSize, scoring=None, loss_func=None, score_func=None,
                                           fit_params=None, n_jobs=1, iid=False, refit=True, n_folds=5, n_repetitions=5,
                                          verbose=0, pre_dispatch='2*n_jobs', random_state=None):



    #initializaiton of random state to a value for each repetition randomly-initialized RandomState object is returned
    #if equal to "None"
    random_state=None

    cv_grid_scores=np.zeros((gridSize, n_repetitions))

    for i in range(0, n_repetitions):
        cv = KFold(X.shape[0], n_folds, shuffle=True, random_state=random_state)
        if(i==0):
            gridSearchCV = GridSearchCV(estimator, param_grid, scoring, loss_func, score_func,
                                fit_params, n_jobs, iid, refit, cv, verbose, pre_dispatch)
        gridSearchCV.set_params(cv=cv)
        gridSearchCV.fit(X, y)
        #gridSearchCV[:,1] returns the mean score for each grid search value
        cv_grid_scores[:, i] = np.array(gridSearchCV.grid_scores_)[:, 1]
    #return average and std_dev of scores across the repetitions of CV
    return cv_grid_scores.mean(axis=1), cv_grid_scores.std(axis=1)



