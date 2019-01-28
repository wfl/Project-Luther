#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: feature_engineering.py
Purpose: 
    1) find a best set of features
    2) find a linear regression that is stable, which means it doesn't
        yield strange predictions
Tools: Pandas, Numpy, seaborn, matplotlib, sklearn, and statmodels
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.graphics import tsaplots


def rsme(y_true, y_pred):
    '''
    This function calculates the root mean square error
    Input arguments:
        y_true = the target's observation
        y_pred = the predicted or estimated target values
    '''
    return np.sqrt(np.mean((y_pred - y_true)**2)) 


# ****** Linear Regression and Standard Scalar ***********

# From Lecture NOTES:
def split_and_validate_LinearR(X_train, X_val, y_train, y_val, \
                               features, has_intercept):
    '''
    This function fits the Linear Regression model with training set, then 
    evaluate the goodness-of-fit using the validation set. Several graphs
    are plotted to visualize the how well the model fit the training set, 
    specifically looks for trends or strange pattern on the residual plots.
    
    Input arguments:
     X_train, y_train = training sets
     X_val, y_val = validation sets
     features = a set of selected features
     has_intercept = a boolean argument. Indicate if the model should
         have an intercept
         
    Output (only for validation set) includes 
     - printed R-square, adjusted R-square, and RSME
    - plot of predicted target Vs actual target
    - plot residuals against these variables (i.e., predicted target, features)
    - qqplot for the residuals
    - autocorrelation plot for the residuals
    - display the regression's coeeficients
    '''
    # fit linear regression to training data
    lr_model = LinearRegression(fit_intercept=has_intercept)
    lr_model.fit(X_train, y_train)
    
    # score fit model on validation data
    val_rsquare = lr_model.score(X_val, y_val)
    val_adj_rsquare = 1 - (1-val_rsquare)*(len(y_val)-1)/(len(y_val)-X_val.shape[1]-1)
    val_yhat = lr_model.predict(X_val)
    val_rmse = rsme(val_yhat, y_val)
    
    # Report R-square, adjusted R-square, and RSME for the validation set
    print('\nLinear: Validation R^2 score was:', val_rsquare)
    print('\nLinear: Validation Adjusted R^2 score was:', val_adj_rsquare)
    print('\nLineat: Validation RMSE was:', val_rmse)    
    
    # Plot: predicted target vs actual target for the validation set
    plt.scatter(val_yhat, y_val.values, color='g')
    plt.xlabel('y', size=12)
    plt.ylabel('y_hat', size=12)
    plt.title('Linear Regression: Predicted Vs Actual for validation data')
    plt.show()
    
    # Plot: residuals vs predicted target for the validation set
    res = y_val.values - val_yhat
    plt.scatter(val_yhat, res, color='g')
    plt.axhline(0,color='r', linestyle='--')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('Residuals', size=12)
    plt.title('Linear Regression: Predicted Vs Residuals for validation data')   
    plt.show()
    
    # Plot: qqplot for the residuals
    sm.qqplot(res, color='g', line='r')   
    plt.show()
    
    # Plot: autocorrelation plot for the residuals
    tsaplots.plot_acf(res, color='g')    
    plt.show()
    
    # Plot: residuals Vs each feature for the validation set
    if type(X_val) is pd.core.series.Series or type(X_val) is pd.core.frame.DataFrame:
        Xplot_df = X_val.copy()
    else:
        Xplot_df = pd.DataFrame(data=X_val[:,0:len(features)-1], columns=features[1:])
    Xplot_df['res'] = res
    sns.pairplot(data=Xplot_df,
                 y_vars=['res'],
                 x_vars=features[1:])
    plt.show()
    
    if has_intercept:
        print('intercept', lr_model.intercept_)
    #Plot: a bar graph to visualize the regression coeeficients    
    fea_len = len(features)
    coefs = pd.Series(lr_model.coef_[0:fea_len-1], features[1:]).sort_values()
    coefs.plot(kind='bar', title="Linear Regression's Coefficients")
    plt.show()


# ****** Regularization and Standard Scalar ***********

def split_and_validate_Regularization(X_train, X_val, y_train, y_val, \
                                      features, has_intercept):
    '''
    This function fits the Lasso and Ridge Regression models with 
    training set, then evaluate the goodness-of-fit using the validation set. 
    Several graphs are plotted to visualize the how well the model fit the 
    training set, specifically looks for trends or strange pattern on the 
    residual plots.
    
    Input arguments:
     X_train, y_train = training sets
     X_val, y_val = validation sets
     features = a set of selected features
     has_intercept = a boolean argument. Indicate if the model should
         have an intercept
         
    Output (only for validation set) includes 
     - printed R-square, adjusted R-square, and RSME
    - plot of predicted target Vs actual target
    - plot residuals against these variables (i.e., predicted target, features)
    - qqplot for the residuals
    - autocorrelation plot for the residuals
    - display the regression's coeeficients
    '''
    lasso_model = Lasso(alpha = 0.0033, fit_intercept=has_intercept) # old 0.01
    ridge_model = Ridge(alpha = 3.1578, normalize=True, fit_intercept=has_intercept) #0.3

    lasso_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    
    # ************************************* 
    # *************** LASSO ***************
    # *************************************     
    # LASSO: core fit model on validation data
    val_rsquare = lasso_model.score(X_val, y_val)
    val_adj_rsquare = 1 - (1-val_rsquare)*(len(y_val)-1)/(len(y_val)-X_val.shape[1]-1)
    val_yhat = lasso_model.predict(X_val)
    val_rmse = rsme(val_yhat, y_val)

    # Report R-square, adjusted R-square, and RSME for the validation set
    print('REGULARIZATION - LASSO')
    print('\nLasso: Validation R^2 score was:', val_rsquare)
    print('\nLasso: Validation Adjusted R^2 score was:', val_adj_rsquare)
    print('\nLasso: Validation RMSE was:', val_rmse)
    
    # Plot: predicted target vs actual target for the validation set
    plt.scatter(val_yhat, y_val.values, color='g')
    plt.xlabel('y', size=12)
    plt.ylabel('y_hat', size=12)
    plt.title('Lasso: Predicted Vs Actual for validation data')
    plt.show()
    
    # Plot: residuals vs predicted target for the validation set
    res = y_val.values - val_yhat
    plt.scatter(val_yhat, res, color='g')
    plt.axhline(0,color='r', linestyle='--')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('Residuals', size=12)
    plt.title('Lasso: Predicted Vs Residuals for validation data')
    plt.show()
 
    # Plot: qqplot for the residuals
    sm.qqplot(res, color='g', line='r')
    plt.show()

    # Plot: autocorrelation plot for the residuals    
    tsaplots.plot_acf(res, color='g')  
    plt.show()
    
    # Plot: residuals Vs each feature for the validation set
    if type(X_val) is pd.core.series.Series or type(X_val) is pd.core.frame.DataFrame:
        Xplot_df = X_val.copy()
    else:
        Xplot_df = pd.DataFrame(data=X_val[:,0:len(features)-1], columns=features[1:])
    Xplot_df['res'] = res
    sns.pairplot(data=Xplot_df,
                 y_vars=['res'],
                 x_vars=features[1:])
    plt.show()

    if has_intercept:
        print('intercept', lasso_model.intercept_)
    #Plot: a bar graph to visualize the regression coeeficients 
    fea_len = len(features)
    coefs = pd.Series(lasso_model.coef_[0:fea_len-1], features[1:]).sort_values()
    coefs.plot(kind='bar', title="Lasso's Coefficients")
    plt.show()

    # ************************************* 
    # *************** RIDGE ***************
    # *************************************     
    val_rsquare = ridge_model.score(X_val, y_val)
    val_adj_rsquare = 1 - (1-val_rsquare)*(len(y_val)-1)/(len(y_val)-X_val.shape[1]-1)
    val_yhat = ridge_model.predict(X_val)
    val_rmse = rsme(val_yhat, y_val)
 
    
    # Report R-square, adjusted R-square, and RSME for the validation set    # report results
    print('REGULARIZATION - RIDGE')
    print('\nRidge: Validation R^2 score was:', val_rsquare)
    print('\nRidge: Validation Adjusted R^2 score was:', val_adj_rsquare)
    print('\nRidge: Validation RMSE was:', val_rmse)

    
    # Plot: predicted target vs actual target for the validation set
    plt.scatter(val_yhat, y_val.values, color='g')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('y_hat (Predicted)',size=12)
    plt.title('Ridge: Predicted Vs Actual for validation data')
    plt.show()
    
    # Plot: residuals vs predicted target for the validation set
    res = y_val.values - val_yhat
    plt.scatter(val_yhat, res, color='g')
    plt.axhline(0,color='r', linestyle='--')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('Residuals', size=12)
    plt.title('Ridge: Predicted Vs Residuals for validation data')  
    plt.show()

    # Plot: qqplot for the residuals
    sm.qqplot(res, color='g', line='r')    
    plt.show()
    
    # Plot: autocorrelation plot for the residuals
    tsaplots.plot_acf(res, color='g')     
    plt.show()

    # Plot: residuals Vs each feature for the validation set
    if type(X_val) is pd.core.series.Series or type(X_val) is pd.core.frame.DataFrame:
       Xplot_df = X_val.copy()
    else:
        Xplot_df = pd.DataFrame(data=X_val[:,0:len(features)-1], columns=features[1:])
    Xplot_df['res'] = res
    sns.pairplot(data=Xplot_df,
                 y_vars=['res'],
                 x_vars=features[1:])
    plt.show()

    if has_intercept:
        print('intercept', ridge_model.intercept_)

    #Plot: a bar graph to visualize the regression coeeficients 
    fea_len = len(features)
    coefs = pd.Series(ridge_model.coef_[0:fea_len-1], features[1:]).sort_values()
    coefs.plot(kind='bar', title="Ridge's Coefficients")
    plt.show()

# *****************************************************************************
# Feature engineering procedue is done by running part of the codes for a set 
#   for a set of feature and look at evaluate the results and plots. 
#   Then, continue to the next set of features.   
# *****************************************************************************
# Load the pickled DataFrame
df = pd.read_pickle('country_14features.pickle')
features = ['code', 'A', 'V', 'P', 'E', 'Gc', 'Cpl', 'R', 'F', 'M', 'Dl', 'Gl', 'Tavg', 'Ravg']

# Extract the target, y
print('Target = cereal yield data')
y = df['Ckg']    

# ----------------------------------------------------------------------------
print('FIRST ATTEMPT: Smaller set of features')
# ----------------------------------------------------------------------------
df_sel = df.loc[:,features[0:8]]
X = pd.get_dummies(df_sel, prefix='code', columns=['code'])

# perform train/val/test split
X_train_val, X_test, y_train_val, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)
    
X_train, X_val, y_train, y_val = \
    train_test_split(X_train_val, y_train_val, test_size=.25, random_state=43)

# Apply Standard Scaler to training set
std = StandardScaler()
std.fit(X_train.values)    
X_train_scaled = std.transform(X_train.values)
#Apply the scaler to the test set
X_val_scaled = std.transform(X_val.values)
X_test_scaled = std.transform(X_test.values)

split_and_validate_LinearR(X_train_scaled, X_val_scaled, y_train, y_val, \
                           features, has_intercept=True)
split_and_validate_Regularization(X_train_scaled, X_val_scaled, y_train, \
                                  y_val, features, has_intercept=True)

# ----------------------------------------------------------------------------
print('SECOND ATTEMPT: include more features')
# ----------------------------------------------------------------------------
df_sel = df.loc[:,features[0:12]]
X = pd.get_dummies(df_sel, prefix='code', columns=['code'])

# perform train/val/test split
X_train_val, X_test, y_train_val, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)
    
X_train, X_val, y_train, y_val = \
    train_test_split(X_train_val, y_train_val, test_size=.25, random_state=43)

# Apply Standard Scaler to training set
std = StandardScaler()
std.fit(X_train.values)    
X_train_scaled = std.transform(X_train.values)
#Apply the scaler to the test set
X_val_scaled = std.transform(X_val.values)
X_test_scaled = std.transform(X_test.values)

split_and_validate_LinearR(X_train_scaled, X_val_scaled, y_train, y_val, \
                           features, has_intercept=True)
split_and_validate_Regularization(X_train_scaled, X_val_scaled, y_train, \
                                  y_val, features, has_intercept=True)

# ----------------------------------------------------------------------------
print('THIRD ATTEMPT: all features')
# ----------------------------------------------------------------------------
df_sel = df.loc[:,features]
X = pd.get_dummies(df_sel, prefix='code', columns=['code'])

# perform train/val/test split
X_train_val, X_test, y_train_val, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)
    
X_train, X_val, y_train, y_val = \
    train_test_split(X_train_val, y_train_val, test_size=.25, random_state=43)

# Apply Standard Scaler to training set
std = StandardScaler()
std.fit(X_train.values)    
X_train_scaled = std.transform(X_train.values)
#Apply the scaler to the test set
X_val_scaled = std.transform(X_val.values)
X_test_scaled = std.transform(X_test.values)

split_and_validate_LinearR(X_train_scaled, X_val_scaled, y_train, y_val, \
                           features, has_intercept=True)
split_and_validate_Regularization(X_train_scaled, X_val_scaled, y_train, \
                                  y_val, features, has_intercept=True)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
print("\nTarget = log-transformed of the cereal yield data")
# apply log-transform to the target y
y = pd.Series(np.log(df['Ckg']))
# ----------------------------------------------------------------------------
print('LOG: Include all features')
# ----------------------------------------------------------------------------
df_sel = df.loc[:,features]
X = pd.get_dummies(df_sel, prefix='code', columns=['code'])

# perform train/val/test split
X_train_val, X_test, y_train_val, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)
    
X_train, X_val, y_train, y_val = \
    train_test_split(X_train_val, y_train_val, test_size=.25, random_state=43)

# Apply Standard Scaler to training set
std = StandardScaler()
std.fit(X_train.values)    
X_train_scaled = std.transform(X_train.values)
#Apply the scaler to the test set
X_val_scaled = std.transform(X_val.values)
X_test_scaled = std.transform(X_test.values)

split_and_validate_LinearR(X_train_scaled, X_val_scaled, y_train, y_val, \
                           features, has_intercept=True)
split_and_validate_Regularization(X_train_scaled, X_val_scaled, y_train, \
                                  y_val, features, has_intercept=True)



'''
# ************************************************************************
#   Consider replacing the categorical variables (country) with 
#       country's centroid location (latitude and longitude)
# Results: lower R-squares  - Need investigation
#************************************************************************
df_lat = df = pd.read_pickle('country_14features_latlong.pickle')
features = ['long','lat', 'A', 'V', 'P', 'E', 'Gc', 'Cpl', 'R', 'F', 'M', 'Dl', 'Gl', 'Tavg', 'Ravg']
df_sel = df.loc[:,features]
X = df.loc[:,features]

# perform train/val/test split
X_train_val, X_test, y_train_val, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)
    
X_train, X_val, y_train, y_val = \
    train_test_split(X_train_val, y_train_val, test_size=.25, random_state=43)

# Apply Standard Scaler to training set
std = StandardScaler()
std.fit(X_train.values)    
X_train_scaled = std.transform(X_train.values)
#Apply the scaler to the test set
X_val_scaled = std.transform(X_val.values)
X_test_scaled = std.transform(X_test.values)

split_and_validate_LinearR(X_train_scaled, X_val_scaled, y_train, y_val, \
                           features, has_intercept=True)
split_and_validate_Regularization(X_train_scaled, X_val_scaled, y_train, \
                                  y_val, features, has_intercept=True)
'''