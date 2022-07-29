import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from rfe_cv import rfe_cv

def main():
  n_samples = 200
  n_features = 9

  rng = np.random.RandomState(0)

  X, y = make_regression(n_samples, n_features, random_state=rng)

  regr = RandomForestRegressor(max_depth=2, 
                               n_estimators=100, 
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



if __name__ == "__main__":
  main()
  

