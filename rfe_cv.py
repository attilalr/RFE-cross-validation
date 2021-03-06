# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import t

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

from sklearn.base import is_classifier
from sklearn.metrics import check_scoring, make_scorer

from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, check_cv, cross_val_score


def rfe_cv(df, vars_x, var_y, estimator, cv=5, scoring='accuracy', std_scaling=False, figsize=(8,4)):

    assert isinstance(df, pd.DataFrame), 'df must be pandas dataframe'

    # this error catch is ugly but I did not found a score <--> estimator check function yet
    try:
      score_vec = cross_val_score(
          estimator, 
          df[vars_x],
          df[var_y],
          scoring=scoring,
          cv=2,
      )
    except:
      raise ValueError('Error. Check if the scoring matches the estimator Regression/Classification case.')


    kfold = check_cv(cv=cv, y=df[var_y], classifier=is_classifier(estimator))
    n_splits = kfold.n_splits

    model = dict()
    model['scores_mean'] = list()
    model['scores_std'] = list()
    model['features_ranking'] = list()
    

    ################### START CV ###

    if is_classifier(estimator):
        y_ = df[var_y]
    else:
        y_ = None

    for i, (train_index, test_index) in enumerate(kfold.split(df, y=y_)):

        if len(vars_x) == 1:
            X_train = df[vars_x].values[train_index].reshape(-1, 1).copy()
            X_test = df[vars_x].values[test_index].reshape(-1, 1).copy()
        else:
            X_train = df[vars_x].values[train_index].copy() # copy of X_train because we are gonna change it
            X_test = df[vars_x].values[test_index].copy() # the same

        y_train = df[var_y].values[train_index] 
        y_test = df[var_y].values[test_index]

        # standard scaling if aplicable
        if std_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        selector = RFE(estimator, n_features_to_select=1, step=1)
        selector = selector.fit(X_train, y_train)

        if i==0: # create new list
            model['features_ranking'].append(list())
            
        model['features_ranking'][-1].append(selector.ranking_) # put value in the last list
    
    ################### END CV ###

    alpha = 0.3173 # for 1 stddev 68,37% CI
    degrees_of_freedom = n_splits - 1
    

    # mean rank figure
    fig1, ax1 = plt.subplots(figsize=figsize)

    v_rank_mean = np.mean(model['features_ranking'], axis=1).ravel()
    v_rank_std = np.mean(model['features_ranking'], axis=1).ravel()

    ax1.plot(v_rank_mean, 'o-')

    ax1.fill_between(
        np.arange(len(v_rank_mean)),
        v_rank_mean + t.ppf(1-alpha/2, degrees_of_freedom)*v_rank_std/np.sqrt(n_splits), # stddev correction using student-t
        v_rank_mean - t.ppf(1-alpha/2, degrees_of_freedom)*v_rank_std/np.sqrt(n_splits),
        alpha=0.1,
        color='b'
    )

    ax1.set_xticks(np.arange(len(v_rank_mean)))
    ax1.set_xticklabels(vars_x, rotation=-40)

    ax1.set_title(f'mean ranking for each feature when modeling {var_y}')

    plt.show()





    # in this loop we are gonna evaluate a score using the i+1 best features
    # using cross_val_score to evaluate
    for i in range(len(vars_x)):

        # rotina pra pegar a n vari??veis mais importantes

        # v ?? o ranking geral, o vetor soma dos rankings em todos cv
        n_best_features = i+1

        v = np.sum(model['features_ranking'], axis=1).ravel()

        # very dirty trick here, to eliminate tied results
        v = v + np.random.random(v.size)/100

        v_sorted = sorted(list(v))
        v_temp_max = v_sorted[n_best_features-1]

        # vamos ver quantas entradas tem com esse valor
        # se tiver 1, o algoritmo ?? mais simples
        if (np.array(v) == v_temp_max).sum() == 1:

            # vetor booleano, True se a coluna pertencer ??s n melhores features
            mask_n_best_features = v <= v_temp_max

        # se tiver mais que 1 precisamos pegar os primeiros
        elif (np.array(v) == v_temp_max).sum() > 1:
            faltam = n_best_features - (np.array(v) < v_temp_max).sum()

            bool_vec1 = np.logical_and(np.array(v) == v_temp_max, 
                                    (np.array(v) == v_temp_max).cumsum() <= faltam)

            mask_n_best_features = np.logical_or(bool_vec1, np.array(v) < v_temp_max)

        else:
            print ('Error')


        # acesso as n melhores features com
        #df[var_x].values[:, mask_n_best_features]
        # ou pra permanecer com o dataframe
        
        selected_col_names = np.array(vars_x)[mask_n_best_features]
        print (f'{i+1} selected features: {selected_col_names}')

        kfold = check_cv(cv=cv, y=df[var_y], classifier=is_classifier(estimator))

        # cross-validated score
        score_vec = cross_val_score(
            estimator, 
            df[selected_col_names],
            df[var_y],
            scoring=scoring,
            cv=kfold,
        )

        model['scores_mean'].append(score_vec.mean())
        model['scores_std'].append(score_vec.std())

    fig2, ax2 = plt.subplots(figsize=figsize)

    ax2.plot(np.arange(len(model['scores_mean']))+1, model['scores_mean'], 'o-')

    ax2.fill_between(
        np.arange(len(model['scores_mean']))+1,
        model['scores_mean'] + t.ppf(1-alpha/2, degrees_of_freedom)*np.array(model['scores_std'])/np.sqrt(n_splits),
        model['scores_mean'] - t.ppf(1-alpha/2, degrees_of_freedom)*np.array(model['scores_std'])/np.sqrt(n_splits),
        alpha=0.1,
        color='b'
    )


    ax2.locator_params(axis='x', nbins=len(model['scores_mean']))
    ax2.set_title(f'modeling {var_y}')
    ax2.set_ylabel(str(scoring))
    ax2.set_xlabel('number of best features')
    #ax2.legend()
    plt.show()


if __name__ == "__main__":
  print (''' - Module rfe-cv -
  
 Use:
 import rfe_cv
 and call rfe_cv.rfe_cv()
 
                 or
 from rfe_cv import rfe_cv 
 and call rfe_cv()
        ''')
