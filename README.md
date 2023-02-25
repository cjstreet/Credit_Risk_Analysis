# Credit_Risk_Analysis
Evaluate three machine learning models by using resampling to determine which is better at predicting credit risk.



## Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. The objective is to use the credit card credit dataset from LendingClub, a peer-to-peer lending services company to develop supervised learning models with unbalanced classes. These various models will be evaluated for performance on reducing bias and predicting credit risk. A final recommendation will be made.


Overview of Steps: 
1. Oversample the data using the RandomOverSampler and SMOTE algorithms
2. Undersample the data using the ClusterCentroids algorithm. 
3. Use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. 
4. Compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 
5. Evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.



## Resources

* Data Source: LoanStats_2019Q1.csv
* ML Libraries: imbalanced-learn, scikit-learn
* Python 3.0
* Jupyter NB

## Results: 
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

## Summary: 
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
