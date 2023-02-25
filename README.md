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
* Python 3.11
* Jupyter NB

## Results: 

The table below contains the balanced accuracy scores, the precision and recall scores of all six machine learning models. The entire Jupyter NB follows the Summary.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>RandomOverSampler</th>
      <th>SMOTE</th>
      <th>ClusterCentroids</th>
      <th>SMOTEEN</th>
      <th>BalancedRandomForestClassifier</th>
      <th>EasyEnsembleClassifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Accuracy</th>
      <td>0.65</td>
      <td>0.62</td>
      <td>0.53</td>
      <td>0.64</td>
      <td>0.78</td>
      <td>0.78</td>
    </tr>
    <tr>
    <th>Precision</th>
      <td>0.99, 0.01, 1.00</td>
      <td>0.99, 0.01, 1.00</td>
      <td>0.99, 0.01, 1.00</td>
      <td>0.99, 0.01, 1.00</td>
      <td>0.99, 0.04, 1.00</td>
      <td>0.99, 0.04, 1.00</td>
    </tr>
    <tr>
   <th>Recall</th>
      <td>0.68</td>
      <td>0.64</td>
      <td>0.45</td>
      <td>0.58</td>
      <td>0.89</td>
      <td>0.89</td>
    </tr>
  </tbody>
</table>
</div>

The credit card data set from LendingClub has the following unbalanced target values:

**low_risk**     68470 : 1
**high_risk**      347 : 0

With each learning model, the resampled dataset was used to train the model, make predictions, and evaluate the model's performance.

Recall is from all the positive classes, how many we predicted correctly.
Precision is from all the classes we have predicted as positive, how many are actually positive.
Accuracy is from all the classes (positive and negative), how many of them we have predicted correctly.



## Summary: 
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.


# Resampling Models to Predict Credit Risk 


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
```

## Read the CSV and Perform Basic Data Cleaning


```python
columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]

target = ["loan_status"]
```


```python
# Load the data
file_path = Path('LoanStats_2019Q1.csv')
df = pd.read_csv(file_path, skiprows=1)[:-2]
df = df.loc[:, columns].copy()

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')

# Drop the null rows
df = df.dropna()

# Remove the `Issued` loan status
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# convert interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100


# Convert the target column values to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}   
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)

df.reset_index(inplace=True, drop=True)

df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>dti</th>
      <th>...</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
      <th>hardship_flag</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10500.0</td>
      <td>0.1719</td>
      <td>375.35</td>
      <td>RENT</td>
      <td>66000.0</td>
      <td>Source Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>27.24</td>
      <td>...</td>
      <td>85.7</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>65687.0</td>
      <td>38199.0</td>
      <td>2000.0</td>
      <td>61987.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25000.0</td>
      <td>0.2000</td>
      <td>929.09</td>
      <td>MORTGAGE</td>
      <td>105000.0</td>
      <td>Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>20.23</td>
      <td>...</td>
      <td>91.2</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>271427.0</td>
      <td>60641.0</td>
      <td>41200.0</td>
      <td>49197.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000.0</td>
      <td>0.2000</td>
      <td>529.88</td>
      <td>MORTGAGE</td>
      <td>56000.0</td>
      <td>Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>24.26</td>
      <td>...</td>
      <td>66.7</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60644.0</td>
      <td>45684.0</td>
      <td>7500.0</td>
      <td>43144.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>0.1640</td>
      <td>353.55</td>
      <td>RENT</td>
      <td>92000.0</td>
      <td>Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>31.44</td>
      <td>...</td>
      <td>100.0</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>99506.0</td>
      <td>68784.0</td>
      <td>19700.0</td>
      <td>76506.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22000.0</td>
      <td>0.1474</td>
      <td>520.39</td>
      <td>MORTGAGE</td>
      <td>52000.0</td>
      <td>Not Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>18.76</td>
      <td>...</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>219750.0</td>
      <td>25919.0</td>
      <td>27600.0</td>
      <td>20000.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>



## Split the Data into Training and Testing

 ### Separate the Features (X) from the Target (y)


```python
# Create our target
y = df.loan_status

# Create our features
X = df.drop(columns="loan_status")

# Create the training variables by converting the string values into numerical ones
X = pd.get_dummies(X)
```


```python
X.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>...</th>
      <th>issue_d_Mar-2019</th>
      <th>pymnt_plan_n</th>
      <th>initial_list_status_f</th>
      <th>initial_list_status_w</th>
      <th>next_pymnt_d_Apr-2019</th>
      <th>next_pymnt_d_May-2019</th>
      <th>application_type_Individual</th>
      <th>application_type_Joint App</th>
      <th>hardship_flag_N</th>
      <th>debt_settlement_flag_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>6.881700e+04</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>...</td>
      <td>68817.000000</td>
      <td>68817.0</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.0</td>
      <td>68817.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16677.594562</td>
      <td>0.127718</td>
      <td>480.652863</td>
      <td>8.821371e+04</td>
      <td>21.778153</td>
      <td>0.217766</td>
      <td>0.497697</td>
      <td>12.587340</td>
      <td>0.126030</td>
      <td>17604.142828</td>
      <td>...</td>
      <td>0.177238</td>
      <td>1.0</td>
      <td>0.123879</td>
      <td>0.876121</td>
      <td>0.383161</td>
      <td>0.616839</td>
      <td>0.860340</td>
      <td>0.139660</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10277.348590</td>
      <td>0.048130</td>
      <td>288.062432</td>
      <td>1.155800e+05</td>
      <td>20.199244</td>
      <td>0.718367</td>
      <td>0.758122</td>
      <td>6.022869</td>
      <td>0.336797</td>
      <td>21835.880400</td>
      <td>...</td>
      <td>0.381873</td>
      <td>0.0</td>
      <td>0.329446</td>
      <td>0.329446</td>
      <td>0.486161</td>
      <td>0.486161</td>
      <td>0.346637</td>
      <td>0.346637</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
      <td>0.060000</td>
      <td>30.890000</td>
      <td>4.000000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9000.000000</td>
      <td>0.088100</td>
      <td>265.730000</td>
      <td>5.000000e+04</td>
      <td>13.890000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>6293.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>15000.000000</td>
      <td>0.118000</td>
      <td>404.560000</td>
      <td>7.300000e+04</td>
      <td>19.760000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>12068.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>24000.000000</td>
      <td>0.155700</td>
      <td>648.100000</td>
      <td>1.040000e+05</td>
      <td>26.660000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>21735.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>40000.000000</td>
      <td>0.308400</td>
      <td>1676.230000</td>
      <td>8.797500e+06</td>
      <td>999.000000</td>
      <td>18.000000</td>
      <td>5.000000</td>
      <td>72.000000</td>
      <td>4.000000</td>
      <td>587191.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 95 columns</p>
</div>




```python
y.head()
```




    0    low_risk
    1    low_risk
    2    low_risk
    3    low_risk
    4    low_risk
    Name: loan_status, dtype: object




```python
# Check the balance of our target values
y.value_counts()
```




    low_risk     68470
    high_risk      347
    Name: loan_status, dtype: int64



**The dataset is unbalanced.**


```python
# # Use the Sklearn `train_test_split()` function to split the data into training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=1, 
                                                    stratify=y)
X_train.shape
```




    (51612, 95)



## Oversampling

In this section, you will compare two oversampling algorithms to determine which algorithm results in the best performance. You will oversample the data using the naive random oversampling algorithm and the SMOTE algorithm. For each algorithm, be sure to complete the folliowing steps:

1. View the count of the target classes using `Counter` from the collections library. 
3. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Print the confusion matrix from sklearn.metrics.
5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.

Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

### Naive Random Oversampling


```python
# confirm the imbalance in the training set
Counter(y_train)
```




    Counter({'low_risk': 51352, 'high_risk': 260})




```python
# Resample the training data with the RandomOversampler

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)
```




    Counter({'low_risk': 51352, 'high_risk': 51352})



**The target dataset is balanced.**


```python
# Train the Logistic Regression model using the resampled data

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score

from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.6456130066757718




```python
# Display the confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
```




    array([[   53,    34],
           [ 5443, 11675]])




```python
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.61      0.68      0.02      0.64      0.41        87
       low_risk       1.00      0.68      0.61      0.81      0.64      0.42     17118
    
    avg / total       0.99      0.68      0.61      0.81      0.64      0.42     17205
    


### SMOTE Oversampling


```python
# Resample the training data with SMOTE
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=1,
sampling_strategy='auto').fit_resample(
   X_train, y_train)
Counter(y_resampled)
```




    Counter({'low_risk': 51352, 'high_risk': 51352})




```python
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.6234433606890912




```python
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
```




    array([[   53,    34],
           [ 6202, 10916]])




```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.61      0.64      0.02      0.62      0.39        87
       low_risk       1.00      0.64      0.61      0.78      0.62      0.39     17118
    
    avg / total       0.99      0.64      0.61      0.77      0.62      0.39     17205
    


## Undersampling

In this section, you will test an undersampling algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above. You will undersample the data using the Cluster Centroids algorithm and complete the folliowing steps:

1. View the count of the target classes using `Counter` from the collections library. 
3. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Print the confusion matrix from sklearn.metrics.
5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.

Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests


```python
# Resample the data using the ClusterCentroids resampler
# Warning: This is a large dataset, and this step may take some time to complete
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
```




    Counter({'high_risk': 260, 'low_risk': 260})




```python
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.5293026900499977




```python
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
```




    array([[  53,   34],
           [9425, 7693]])




```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.61      0.45      0.01      0.52      0.28        87
       low_risk       1.00      0.45      0.61      0.62      0.52      0.27     17118
    
    avg / total       0.99      0.45      0.61      0.62      0.52      0.27     17205
    


## Combination (Over and Under) Sampling

In this section, you will test a combination over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above. You will resample the data using the SMOTEENN algorithm and complete the folliowing steps:

1. View the count of the target classes using `Counter` from the collections library. 
3. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Print the confusion matrix from sklearn.metrics.
5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.

Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests


```python
# Resample the training data with SMOTEENN

from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
Counter(y_resampled)
```




    Counter({'high_risk': 68460, 'low_risk': 62011})




```python
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.6395687540036501




```python
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
```




    array([[  61,   26],
           [7224, 9894]])




```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.70      0.58      0.02      0.64      0.41        87
       low_risk       1.00      0.58      0.70      0.73      0.64      0.40     17118
    
    avg / total       0.99      0.58      0.70      0.73      0.64      0.40     17205
    

# Ensemble Classifiers to Predict Credit Risk


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
```


```python
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
```

## Read the CSV and Perform Basic Data Cleaning


```python
# https://help.lendingclub.com/hc/en-us/articles/215488038-What-do-the-different-Note-statuses-mean-

columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]

target = ["loan_status"]
```


```python
# Load the data
file_path = Path('LoanStats_2019Q1.csv')
df = pd.read_csv(file_path, skiprows=1)[:-2]
df = df.loc[:, columns].copy()

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')

# Drop the null rows
df = df.dropna()

# Remove the `Issued` loan status
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# convert interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100


# Convert the target column values to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}   
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)

df.reset_index(inplace=True, drop=True)

df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>dti</th>
      <th>...</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
      <th>hardship_flag</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10500.0</td>
      <td>0.1719</td>
      <td>375.35</td>
      <td>RENT</td>
      <td>66000.0</td>
      <td>Source Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>27.24</td>
      <td>...</td>
      <td>85.7</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>65687.0</td>
      <td>38199.0</td>
      <td>2000.0</td>
      <td>61987.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25000.0</td>
      <td>0.2000</td>
      <td>929.09</td>
      <td>MORTGAGE</td>
      <td>105000.0</td>
      <td>Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>20.23</td>
      <td>...</td>
      <td>91.2</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>271427.0</td>
      <td>60641.0</td>
      <td>41200.0</td>
      <td>49197.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000.0</td>
      <td>0.2000</td>
      <td>529.88</td>
      <td>MORTGAGE</td>
      <td>56000.0</td>
      <td>Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>24.26</td>
      <td>...</td>
      <td>66.7</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60644.0</td>
      <td>45684.0</td>
      <td>7500.0</td>
      <td>43144.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>0.1640</td>
      <td>353.55</td>
      <td>RENT</td>
      <td>92000.0</td>
      <td>Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>31.44</td>
      <td>...</td>
      <td>100.0</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>99506.0</td>
      <td>68784.0</td>
      <td>19700.0</td>
      <td>76506.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22000.0</td>
      <td>0.1474</td>
      <td>520.39</td>
      <td>MORTGAGE</td>
      <td>52000.0</td>
      <td>Not Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>18.76</td>
      <td>...</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>219750.0</td>
      <td>25919.0</td>
      <td>27600.0</td>
      <td>20000.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>



## Split the Data into Training and Testing


```python
# Create our target
y = df.loan_status

# Create our features
X = df.drop(columns="loan_status")

# Create the training variables by converting the string values into numerical ones
X = pd.get_dummies(X)
```


```python
X.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>...</th>
      <th>issue_d_Mar-2019</th>
      <th>pymnt_plan_n</th>
      <th>initial_list_status_f</th>
      <th>initial_list_status_w</th>
      <th>next_pymnt_d_Apr-2019</th>
      <th>next_pymnt_d_May-2019</th>
      <th>application_type_Individual</th>
      <th>application_type_Joint App</th>
      <th>hardship_flag_N</th>
      <th>debt_settlement_flag_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>6.881700e+04</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>...</td>
      <td>68817.000000</td>
      <td>68817.0</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.0</td>
      <td>68817.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16677.594562</td>
      <td>0.127718</td>
      <td>480.652863</td>
      <td>8.821371e+04</td>
      <td>21.778153</td>
      <td>0.217766</td>
      <td>0.497697</td>
      <td>12.587340</td>
      <td>0.126030</td>
      <td>17604.142828</td>
      <td>...</td>
      <td>0.177238</td>
      <td>1.0</td>
      <td>0.123879</td>
      <td>0.876121</td>
      <td>0.383161</td>
      <td>0.616839</td>
      <td>0.860340</td>
      <td>0.139660</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10277.348590</td>
      <td>0.048130</td>
      <td>288.062432</td>
      <td>1.155800e+05</td>
      <td>20.199244</td>
      <td>0.718367</td>
      <td>0.758122</td>
      <td>6.022869</td>
      <td>0.336797</td>
      <td>21835.880400</td>
      <td>...</td>
      <td>0.381873</td>
      <td>0.0</td>
      <td>0.329446</td>
      <td>0.329446</td>
      <td>0.486161</td>
      <td>0.486161</td>
      <td>0.346637</td>
      <td>0.346637</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
      <td>0.060000</td>
      <td>30.890000</td>
      <td>4.000000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9000.000000</td>
      <td>0.088100</td>
      <td>265.730000</td>
      <td>5.000000e+04</td>
      <td>13.890000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>6293.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>15000.000000</td>
      <td>0.118000</td>
      <td>404.560000</td>
      <td>7.300000e+04</td>
      <td>19.760000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>12068.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>24000.000000</td>
      <td>0.155700</td>
      <td>648.100000</td>
      <td>1.040000e+05</td>
      <td>26.660000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>21735.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>40000.000000</td>
      <td>0.308400</td>
      <td>1676.230000</td>
      <td>8.797500e+06</td>
      <td>999.000000</td>
      <td>18.000000</td>
      <td>5.000000</td>
      <td>72.000000</td>
      <td>4.000000</td>
      <td>587191.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 95 columns</p>
</div>




```python
# Check the balance of our target values
y.value_counts()
```




    low_risk     68470
    high_risk      347
    Name: loan_status, dtype: int64




```python

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

## Ensemble Learners

In this section, you will compare two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier . For each algorithm, be sure to complete the folliowing steps:

1. Train the model using the training data. 
2. Calculate the balanced accuracy score from sklearn.metrics.
3. Print the confusion matrix from sklearn.metrics.
4. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
5. For the Balanced Random Forest Classifier onely, print the feature importance sorted in descending order (most important feature to least important) along with the feature score

Note: Use a random state of 1 for each algorithm to ensure consistency between tests

### Balanced Random Forest Classifier


```python
# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
random_forest = BalancedRandomForestClassifier(n_estimators = 100)
random_forest = random_forest.fit(X_train, y_train)
```


```python
# Calculated the balanced accuracy score

y_pred = random_forest.predict(X_test)
print(f"Accuracy Score : {balanced_accuracy_score(y_test, y_pred)}")
```

    Accuracy Score : 0.7821434856301346



```python
# Display the confusion matrix

confusion_matrix(y_test, y_pred)
```




    array([[   68,    33],
           [ 1864, 15240]])




```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.04      0.67      0.89      0.07      0.77      0.59       101
       low_risk       1.00      0.89      0.67      0.94      0.77      0.61     17104
    
    avg / total       0.99      0.89      0.67      0.94      0.77      0.61     17205
    



```python
# List the features sorted in descending order by feature importance
# Calculate feature importance in the Random Forest model.
sorted(zip(random_forest.feature_importances_, X.columns), reverse=True)
```




    [(0.07742904150973835, 'total_rec_prncp'),
     (0.06259896859934252, 'total_pymnt_inv'),
     (0.05683561542433521, 'total_pymnt'),
     (0.05331706211791855, 'total_rec_int'),
     (0.052823587106665515, 'last_pymnt_amnt'),
     (0.030099074535899613, 'int_rate'),
     (0.0197653531417215, 'issue_d_Jan-2019'),
     (0.0171937095163081, 'mths_since_recent_inq'),
     (0.01694275684130075, 'dti'),
     (0.016857957140889956, 'out_prncp_inv'),
     (0.016624287664126217, 'out_prncp'),
     (0.015693798598294326, 'installment'),
     (0.015623297779128265, 'annual_inc'),
     (0.015392109699693322, 'mths_since_rcnt_il'),
     (0.01536973313997631, 'revol_bal'),
     (0.014882014277010197, 'mo_sin_old_rev_tl_op'),
     (0.0146303202314117, 'il_util'),
     (0.014380905630456116, 'bc_open_to_buy'),
     (0.014130625255190371, 'max_bal_bc'),
     (0.014090391073322139, 'total_bc_limit'),
     (0.013953764774453264, 'issue_d_Mar-2019'),
     (0.013909986386943105, 'tot_hi_cred_lim'),
     (0.01362238890967919, 'tot_cur_bal'),
     (0.013588714048629607, 'avg_cur_bal'),
     (0.013478602385048192, 'mo_sin_old_il_acct'),
     (0.013271494748576735, 'total_rev_hi_lim'),
     (0.01326494979907962, 'all_util'),
     (0.013263529889268339, 'total_il_high_credit_limit'),
     (0.013134674777429472, 'total_bal_il'),
     (0.01269191036053539, 'mths_since_recent_bc'),
     (0.01268783088722474, 'total_bal_ex_mort'),
     (0.011794821395216409, 'total_acc'),
     (0.0116449862618102, 'bc_util'),
     (0.011202018867384738, 'loan_amnt'),
     (0.010734374266329727, 'mo_sin_rcnt_rev_tl_op'),
     (0.010400139429310704, 'inq_last_12m'),
     (0.010222492994945596, 'acc_open_past_24mths'),
     (0.010039319743215298, 'num_rev_accts'),
     (0.009522566689855278, 'num_il_tl'),
     (0.009444752122501702, 'pct_tl_nvr_dlq'),
     (0.009425037960413173, 'num_sats'),
     (0.008960704299178623, 'num_actv_rev_tl'),
     (0.008758204360573707, 'num_bc_tl'),
     (0.008717715932548673, 'open_il_24m'),
     (0.008701616532463059, 'num_op_rev_tl'),
     (0.008419394874771852, 'num_tl_op_past_12m'),
     (0.008173868100530161, 'inq_fi'),
     (0.008156803855678624, 'num_actv_bc_tl'),
     (0.008084951549152154, 'open_acc'),
     (0.007860432819238525, 'mo_sin_rcnt_tl'),
     (0.007621675606626275, 'num_rev_tl_bal_gt_0'),
     (0.007384428745716938, 'total_rec_late_fee'),
     (0.00726498714005818, 'total_cu_tl'),
     (0.007115284407371268, 'num_bc_sats'),
     (0.007099747014061353, 'open_rv_24m'),
     (0.006515432768595377, 'open_act_il'),
     (0.005815827289974592, 'open_acc_6m'),
     (0.005396014609460902, 'percent_bc_gt_75'),
     (0.004991018810114, 'mort_acc'),
     (0.0049196013316022275, 'inq_last_6mths'),
     (0.004819490357850445, 'open_il_12m'),
     (0.004599462648741739, 'next_pymnt_d_Apr-2019'),
     (0.004498610117692187, 'tot_coll_amt'),
     (0.0043147368482016725, 'open_rv_12m'),
     (0.004199027452284657, 'next_pymnt_d_May-2019'),
     (0.00406938716903378, 'delinq_2yrs'),
     (0.003446397591226702, 'issue_d_Feb-2019'),
     (0.0031769242121804735, 'num_accts_ever_120_pd'),
     (0.0022151813628364546, 'application_type_Joint App'),
     (0.0020585896424842164, 'verification_status_Verified'),
     (0.002047108062125063, 'home_ownership_MORTGAGE'),
     (0.001787277151408717, 'home_ownership_OWN'),
     (0.0017388708376012347, 'verification_status_Not Verified'),
     (0.0017009569665482757, 'application_type_Individual'),
     (0.0016570243770208364, 'verification_status_Source Verified'),
     (0.0016260701110992907, 'pub_rec'),
     (0.0013216449384939265, 'initial_list_status_w'),
     (0.0011925084993531896, 'pub_rec_bankruptcies'),
     (0.0011251541136919343, 'home_ownership_RENT'),
     (0.000943066863061437, 'initial_list_status_f'),
     (0.0009303706626806725, 'num_tl_90g_dpd_24m'),
     (0.0002910325719856567, 'collections_12_mths_ex_med'),
     (0.00021056160308602863, 'chargeoff_within_12_mths'),
     (9.38698110154859e-05, 'home_ownership_ANY'),
     (0.0, 'tax_liens'),
     (0.0, 'recoveries'),
     (0.0, 'pymnt_plan_n'),
     (0.0, 'policy_code'),
     (0.0, 'num_tl_30dpd'),
     (0.0, 'num_tl_120dpd_2m'),
     (0.0, 'hardship_flag_N'),
     (0.0, 'delinq_amnt'),
     (0.0, 'debt_settlement_flag_N'),
     (0.0, 'collection_recovery_fee'),
     (0.0, 'acc_now_delinq')]



### Easy Ensemble AdaBoost Classifier


```python
# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(X_train, y_train)


print(f'Training Score: {eec.score(X_train, y_train)}')
print(f'Testing Score: {eec.score(X_test, y_test)}')

```

    Training Score: 0.9435015112764473
    Testing Score: 0.9433304272013949



```python
# Calculated the balanced accuracy score

y_pred = random_forest.predict(X_test)
acc_score = balanced_accuracy_score(y_test, y_pred)
print(f"Accuracy Score : {acc_score}")
```

    Accuracy Score : 0.7821434856301346



```python
# Display the confusion matrix

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
   cm, index=["Actual 0", "Actual 1"],
   columns=["Predicted 0", "Predicted 1"]
)
display(cm_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted 0</th>
      <th>Predicted 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual 0</th>
      <td>68</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Actual 1</th>
      <td>1864</td>
      <td>15240</td>
    </tr>
  </tbody>
</table>
</div>



```python
# print the imbalanced classification report
print("Imbalanced Classification Report")
print(classification_report_imbalanced(y_test, y_pred))
```

    Imbalanced Classification Report
                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.04      0.67      0.89      0.07      0.77      0.59       101
       low_risk       1.00      0.89      0.67      0.94      0.77      0.61     17104
    
    avg / total       0.99      0.89      0.67      0.94      0.77      0.61     17205
    


