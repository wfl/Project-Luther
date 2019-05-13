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
import pickle

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

import matplotlib.pyplot as plt



def rsme(y_true, y_pred):
    '''
    This function calculates the root mean square error
    Input arguments:
        y_true = the target's observation
        y_pred = the predicted or estimated target values
    '''
    return np.sqrt(np.mean((y_pred - y_true)**2)) 


# ****** Linear Regression ***********

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
    plt.xlabel('y_hat', size=12)
    plt.ylabel('y', size=12)
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

    
    df_res = pd.DataFrame(res, index=y_val.index)
    with open('df_res_temp.pickle', 'wb') as write_to:
        pickle.dump(df_res, write_to) 
             
    # Plot: qqplot for the residuals
    sm.qqplot(res, color='g', line='r')   
    plt.show()
    
    # Plot: autocorrelation plot for the residuals
    tsaplots.plot_acf(res, color='g')    
    plt.show()
    
    '''
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
    '''
        
    if has_intercept:
        print('intercept', lr_model.intercept_)

    #Plot: a bar graph to visualize the regression coeeficients    
    fea_len = len(features)
    
    if 'code' in features: # With dummy variables
        coefs = pd.Series(lr_model.coef_[0:fea_len-1], features[1:]).sort_values()
    else: # No entities and No dummy variables
        coefs = pd.Series(lr_model.coef_, features).sort_values()
    coefs.plot(kind='bar', title="Linear Regression's Coefficients")
    plt.show()


# ****** Regularization ***********

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
    ridge_model = Ridge(alpha = 3.1578, fit_intercept=has_intercept) #0.3
    #ridge_model = Ridge(fit_intercept=has_intercept) #0.3

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
    plt.scatter(val_yhat, y_val.values, color='b')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('y', size=12)
    plt.title('Lasso: Predicted Vs Actual for validation data')
    plt.show()
    
    # Plot: residuals vs predicted target for the validation set
    res = y_val.values - val_yhat
    plt.scatter(val_yhat, res, color='b')
    plt.axhline(0,color='r', linestyle='--')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('Residuals', size=12)
    plt.title('Lasso: Predicted Vs Residuals for validation data')
    plt.show()
 
    # Plot: qqplot for the residuals
    sm.qqplot(res, color='b', line='r')
    plt.show()

    # Plot: autocorrelation plot for the residuals    
    tsaplots.plot_acf(res, color='b')  
    plt.show()
    
    '''
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
    '''
    
    if has_intercept:
        print('intercept', lasso_model.intercept_)
    #Plot: a bar graph to visualize the regression coeeficients 
    fea_len = len(features)
    
    if 'code' in features: # With dummy variables
        coefs = pd.Series(lasso_model.coef_[0:fea_len-1], features[1:]).sort_values()
    else: # No entities and No dummy variables
        coefs = pd.Series(lasso_model.coef_, features).sort_values()

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
    plt.scatter(val_yhat, y_val.values, color='m')
    plt.xlabel('y', size=12)
    plt.ylabel('y_hat',size=12)
    plt.title('Ridge: Predicted Vs Actual for validation data')
    plt.show()
    
    # Plot: residuals vs predicted target for the validation set
    res = y_val.values - val_yhat 
    plt.scatter(val_yhat, res, color='m')
    plt.axhline(0,color='r', linestyle='--')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('Residuals', size=12)
    plt.title('Ridge: Predicted Vs Residuals for validation data')  
    plt.show()

    # Plot: qqplot for the residuals
    sm.qqplot(res, color='m', line='r')    
    plt.show()
    
    # Plot: autocorrelation plot for the residuals
    tsaplots.plot_acf(res, color='m')     
    plt.show()

    '''
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
    '''

    if has_intercept:
        print('intercept', ridge_model.intercept_)

    #Plot: a bar graph to visualize the regression coeeficients 
    fea_len = len(features)
    if 'code' in features: # With dummy variables
        coefs = pd.Series(ridge_model.coef_[0:fea_len-1], features[1:]).sort_values()
    else: # No entities and No dummy variables
        coefs = pd.Series(ridge_model.coef_, features).sort_values()
    
    coefs.plot(kind='bar', title="Ridge's Coefficients")
    plt.show()



if __name__ == "__main__":

    '''
    NOTES: 
    Did not apply standardscalar to panel data. It seems not
    advisable.
    
    LSDV better model but there are outliners that need to be investigated.
    '''
    
    # *****************************************************************************
    # Feature engineering procedue is done by running part of the codes for a set 
    #   for a set of feature and look at evaluate the results and plots. 
    #   Then, continue to the next set of features.   
    # *****************************************************************************
    # Load the pickled DataFrame from the data/processed folder
    df = pd.read_pickle('df_train_clean.pickle')
    
    
    # ----------------------------------------------------------------------------
    # Fixed effects regresion model:  
    #     OLS regression model called least square with dummy variables model
    # ----------------------------------------------------------------------------
    # perform train/val/test split for time series panel data
    features = ['code', 'A', 'V', 'E', 'F', 'M', 'Gc', 'Dl', 'R', 'Tavg', 'Rmin']
    df_X = df[(df['year']>='1991') & (df['year']<='2005')]
    df_X_val = df[(df['year']>='2006') & (df['year']<='2010')]
    
    print('Target = cereal yield data')
    # Extract the target, y
    y_train = df_X['Ckg'] 
    y_val = df_X_val['Ckg']
    
    print('Generate dummy variables for the entities (countries)')
    df_X = df_X.loc[:,features]
    df_X_val = df_X_val.loc[:,features]
    
    # Generate dummy variables for the entities (countries)
    X_train = pd.get_dummies(df_X, prefix='code', columns=['code'])
    X_val = pd.get_dummies(df_X_val, prefix='code', columns=['code'])
    
    X_train_scaled = X_train
    X_val_scaled = X_val
    
    '''
    # Ignore this step: Apply Standard Scaler to training set
    std = StandardScaler()
    std.fit(X_train.values)    
    X_train_scaled = std.transform(X_train.values)
    #Apply the scaler to the test set
    X_val_scaled = std.transform(X_val.values)
    '''
    
    print("LEAST SQUARE WITH DUMMY VARIABLES MODEL")
    split_and_validate_LinearR(X_train_scaled, X_val_scaled, y_train, y_val, \
                               features, has_intercept=True)
    split_and_validate_Regularization(X_train_scaled, X_val_scaled, y_train, \
                                      y_val, features, has_intercept=True)
    
