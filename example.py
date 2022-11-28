import numpy as np
import pandas as pd

from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from rfe_cv import rfe_cv

def main():
  n_samples = 200
  n_features = 6

  rng = np.random.RandomState(0)

  X, y = make_regression(n_samples, n_features, random_state=rng)

  regr = RandomForestRegressor(max_depth=2, 
                               n_estimators=40, 
                               random_state=rng,
                               )


  vars_names = [f'var {x}' for x in range(n_features)]
  df = pd.DataFrame(data=X, columns=vars_names)
  df['y'] = y

  # the scoring parameter is used in the scoring X no. of best features.
  # the RFE phase is performed using sklearn's RFE.
  rfe_cv(df, vars_names, 'y', regr, 
         cv=5,
         scoring='r2', 
         figsize=(7, 4),
         )



  # Lets perform a classification
  X, y = make_classification(n_samples, 
                             n_features, 
                             n_classes=2, 
                             random_state=rng,
                             )

  vars_names = [f'var {x}' for x in range(n_features)]
  df = pd.DataFrame(data=X, columns=vars_names)
  df['y'] = y


  clf = RandomForestClassifier(max_depth=2, 
                               n_estimators=40, 
                               random_state=rng,
                               )


  rfe_cv(df, vars_names, 'y', clf, 
         cv=5,
         scoring='accuracy', 
         figsize=(7, 4),
         )




  # Two models in the same figure
  clf = RandomForestClassifier(max_depth=2, 
                               n_estimators=40, 
                               random_state=rng,
                               )

  fig1, fig2 = rfe_cv(df, vars_names, 'y', clf, 
         cv = 5,
         scoring = 'accuracy', 
         figsize = (7, 4),
         return_fig = True,
         model_label = 'RF1',
         )
         
  clf = RandomForestClassifier(max_depth=None, 
                               n_estimators=10, 
                               random_state=rng,
                               )
  rfe_cv(df, vars_names, 'y', clf, 
         cv = 5,
         scoring = 'accuracy', 
         return_fig = False,
         figs = [fig1, fig2],
         model_label = 'RF2',
         )
         

  # if I dont want all features
  rfe_cv(df, vars_names, 'y', clf, 
         cv = 5,
         scoring = 'accuracy', 
         max_features = 3,
         model_label = 'RF2a',
         return_fig = False,
         )


if __name__ == "__main__":
  main()
  

