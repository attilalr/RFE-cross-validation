# RFE-cross-validation
RFE (Recursive Feature Elimination) with cross validation

RFECV from scikit-learn optimizes the number of best features [1] but is usual the desire to know the selected variables for other cases. The RFE function addresses this issue but is it not cross-validated.

A cross-validation for feature importance can give the idea of how a given feature is well established in the feature importance rank. The rank variability is shown by the standard deviation of the rank mean. As the next figure shows: 

![image](https://user-images.githubusercontent.com/9744889/160865097-9005a1b4-d4f2-4bde-bad6-be07037fe0a7.png)

Figure: Mean rank across folds for each feature.


TO-DO:

- ~~Separate module file and example script;~~
- Prettier output or returning of results;
- Input parameters for figures (done for figsize);
- Get rid of commentaries in portuguese (on the way);
- Return of figures instead of plotting;
- Better check of inputs (types);
- Consider the variability of the std deviation of the mean;
- Richer example.

[1] http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

[2] https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
