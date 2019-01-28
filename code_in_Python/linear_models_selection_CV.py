#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: linear_model_selection_CV.py
Purpose: 
    1) use crossvalidation to find the best alphas for Lasso and Ridge 
        regression models
    2) find the best models among linear regression, Lasso, and Ridge
Tools: Pandas, Numpy, seaborn, matplotlib, and sklearn
References:
    https://stackoverflow.com/questions/46633273/sklearn-kfold-returning-wrong-indexes-in-python
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


# Calculate RMSE
def rsme(y_true, y_pred):
    '''
    This function calculates the root mean square error
    Input arguments:
        y_true = the target's observation
        y_pred = the predicted or estimated target values
    '''    
    return np.sqrt(np.mean((y_pred - y_true)**2)) 

# Calculate adjusted R2
def AdjR2s(X, y, rsquare):
    '''
    This function calculates the adjusted R-square
    Input arguments:
        X = features
        y = target values
        rsquare = R-squares
    '''        
    return 1 - (1-rsquare)*(len(y)-1)/(len(y)-X.shape[1]-1)


def Run_Kfolds(X_train_val, y_train_val, nfolds, alpha_Lasso, alpha_Ridge, \
               has_intercept):
    '''
    This function runs the cross validation procedures to find out
        which model returns the best goodness-of-fit metric
    Input arguments:
        X_train_val, y_train_val = train and validation set
        nfolds = number of folds
        lpha_Lasso, alpha_Ridge = the alphas parameters for Lasso and Ridge
                                  respectively
    has_intercept = a boolean argument. If True, models must include intercept
    '''        
    # set up containers to collect the metrics that are calculated for 
    #   all models from the validation set 
    cv_lr_r2s, cv_lasso_r2s, cv_ridge_r2s  = [], [], [] 
    cv_lr_Ar2s, cv_lasso_Ar2s, cv_ridge_Ar2s  = [], [], [] 
    cv_lr_rsme, cv_lasso_rsme, cv_ridge_rsme  = [], [], [] 
    
    # Start the cross validation procedure
    kf = KFold(n_splits=nfolds, shuffle=False, random_state = 71)
    
    for train_ind, val_ind in kf.split(X_train_val, y_train_val):
        
        if type(X_train_val) is pd.core.series.Series or type(X_train_val) is pd.core.frame.DataFrame:
            X_train, y_train = X_train_val.iloc[train_ind], y_train_val.iloc[train_ind]
            X_val, y_val = X_train_val.iloc[val_ind], y_train_val.iloc[val_ind]
        else:
            X_train, y_train = X_train_val[train_ind], y_train_val[train_ind]
            X_val, y_val = X_train_val[val_ind], y_train_val[val_ind] 
        
        # Apply standardScaler to training set
        std = StandardScaler()
        std.fit(X_train.values)    
        X_train_scaled = std.transform(X_train.values)
        #Apply the scaler to the val and test set
        X_val_scaled = std.transform(X_val.values)
        
        # Create models' instance
        lr_model = LinearRegression(fit_intercept=has_intercept)
        lasso_model = Lasso(alpha_Lasso, fit_intercept=has_intercept)
        ridge_model = Ridge(alpha_Ridge, normalize=True, fit_intercept=has_intercept)
        
        # Fit the model
        lasso_model.fit(X_train_scaled, y_train)
        ridge_model.fit(X_train_scaled, y_train)    
        lr_model.fit(X_train_scaled, y_train)   
        
        # Store r2
        cv_lr_r2s.append(lr_model.score(X_val_scaled, y_val))
        cv_lasso_r2s.append(lasso_model.score(X_val_scaled, y_val))
        cv_ridge_r2s.append(ridge_model.score(X_val_scaled, y_val))
        
        # Store Adjusted r2
        cv_lr_Ar2s.append(AdjR2s(X_val, y_val, cv_lr_r2s[-1]))
        cv_lasso_Ar2s.append(AdjR2s(X_val, y_val, cv_lasso_r2s[-1]))
        cv_ridge_Ar2s.append(AdjR2s(X_val, y_val, cv_ridge_r2s[-1]))
        
        # Store RSME
        val_yhat = lr_model.predict(X_val)
        cv_lr_rsme.append(rsme(val_yhat, y_val) )
        
        val_yhat = lasso_model.predict(X_val) 
        cv_lasso_rsme.append(rsme(val_yhat, y_val))
        
        val_yhat = ridge_model.predict(X_val)
        cv_ridge_rsme.append(rsme(val_yhat, y_val))
        
    # print the results
    print('\nLinear Regression R^2: ', cv_lr_r2s)
    print('\nLasso R^2: ', cv_lasso_r2s)
    print('\nRidge R^2: ', cv_ridge_r2s)

    print('\nLinear Regression: Adjusted R^2: ', cv_lr_Ar2s)
    print('\nLasso Adjusted R^2: ', cv_lasso_Ar2s)
    print('\nRidge Adjusted R^2: ', cv_ridge_Ar2s)
    
    print('\nLinear Regression: RMSE: ', cv_lr_rsme)
    print('\nLasso RSME: ', cv_lasso_rsme)
    print('\nRidge RSME: ', cv_ridge_rsme)
    
    print(f'\nLinear Regression Average R^2: {np.mean(cv_lr_r2s):.3f} +- {np.std(cv_lr_r2s):.3f}')
    print(f'Lasso mean Average R^2: {np.mean(cv_lasso_r2s):.3f} +- {np.std(cv_lasso_r2s):.3f}')    
    print(f'Ridge mean Average R^2: {np.mean(cv_ridge_r2s):.3f} +- {np.std(cv_ridge_r2s):.3f}')    

    print(f'\nLinear Regression Average RMSE: {np.mean(cv_lr_rsme):.3f} +- {np.std(cv_lr_rsme):.3f}')
    print(f'Lasso mean Average RMSE: {np.mean(cv_lasso_rsme):.3f} +- {np.std(cv_lasso_rsme):.3f}')    
    print(f'Ridge mean Average RMSE: {np.mean(cv_ridge_rsme):.3f} +- {np.std(cv_ridge_rsme):.3f}')    
     
    return print('\nCross validation has completed.')
    


def LinearR(X_train_val, y_train_val, X_test, y_test, X, y, features, has_intercept):
    '''
    This function fits the Linear Regression model with training and validation
    sets together,then evaluate the goodness-of-fit using the test set. 
    Several graphs are plotted to visualize the how well the model fit the test
    set.
    
    Input arguments:
     X_train_val, y_train_val = training and validation sets
     X_test, y_test = test sets
     features = a set of selected features 
     has_intercept = a boolean argument. Indicate if the model should
         have an intercept
         
    Output (only for test set) includes 
     - printed R-square, adjusted R-square, and RSME
    - plot of predicted target Vs actual target
    - plot residuals against these variables (i.e., predicted target, features)
    - display the regression's coeeficients
    '''    
    
    # Apply Standard Scaler to training set
    std = StandardScaler()
    std.fit(X_train_val.values)    
    X_train_val_scaled = std.transform(X_train_val.values)
    #Apply the scaler to the test set
    X_test_scaled = std.transform(X_test.values)
    
    
    # # fit linear regression to train+val data
    lr_model = LinearRegression(fit_intercept=has_intercept)
    lr_model.fit(X_train_val_scaled, y_train_val)
    
    # score fit model on test data
    test_rsquare = lr_model.score(X_test_scaled, y_test)
    test_adj_rsquare = 1 - (1-test_rsquare)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    test_yhat = lr_model.predict(X_test_scaled)
    test_rmse = rsme(test_yhat, y_test)
    
    # Report R-square, adjusted R-square, and RSME for the test data 
    print('\nLinear: Test R^2 score was:', test_rsquare)
    print('\nLinear: Test Adjusted R^2 score was:', test_adj_rsquare)
    print('\nLinear: Test RMSE was:', test_rmse)
    
    
    # Plot: predicted target vs actual target for the test set
    plt.scatter(test_yhat, y_test, color='g')
    plt.xlabel('y', size=12)
    plt.ylabel('y_hat', size=12)
    plt.title('Linear Regression: Predicted Vs Actual for TEST data')
    plt.show()
    
    # Plot: residuals vs predicted target for the test set
    res = y_test- test_yhat
    plt.scatter(test_yhat, res, color='g')
    plt.axhline(0,color='r', linestyle='--')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('Residuals', size=12)
    plt.title('Linear Regression: Predicted Vs Residuals for TEST data')   
    plt.show()

    
    if has_intercept:
        print('intercept', lr_model.intercept_)
    #Plot: a bar graph to visualize the regression coeeficients  
    fea_len = len(features)
    coefs = pd.Series(lr_model.coef_[0:fea_len-1], features[1:]).sort_values()
    coefs.plot(kind='bar', title="Linear Regression's Coefficients")    
    plt.show()

    
#********** Training and Testing the models ****************
# Load the pickled DataFrame
df_cat = pd.read_pickle('country_14features.pickle')
# Apply log-transform on the target
y = pd.Series(np.log(df_cat['Ckg'])) 
features = ['code', 'A', 'V', 'P', 'E', 'F', 'M', 'Gc', 'Dl', 'Gl', 'Cpl', 'R', 'Tavg', 'Ravg']
df_cat_sel = df_cat.loc[:,features]
X = pd.get_dummies(df_cat_sel, prefix='code', columns=['code'])

# perform train/val/test split
X_train_val, X_test, y_train_val, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)
    
X_train, X_val, y_train, y_val = \
    train_test_split(X_train_val, y_train_val, test_size=.25, random_state=43)

# Apply Standard Scaler to training set
std = StandardScaler()
std.fit(X_train.values)    
X_train_scaled = std.transform(X_train.values)
#Apply the scaler to the val and test set
X_val_scaled = std.transform(X_val.values)
X_test_scaled = std.transform(X_test.values)

# ----------------------------------------------------------------------  
#   Find the average alpha values for Lasso and Ridge
# ----------------------------------------------------------------------  
nfolds = 10
alphavec = np.linspace(-1,2,300)
lasso_model = LassoCV(alphas = alphavec, cv=nfolds)
lasso_model.fit(X_train_scaled, y_train)
print("CATEGORICAL lasso_model.alpha_ = ", lasso_model.alpha_)
val_set_pred = lasso_model.predict(X_val_scaled)
print("CATEGORICAL Lasso RMSE for Val = ", rsme(y_val, val_set_pred))
test_set_pred = lasso_model.predict(X_test_scaled)
print("CATEGORICAL Lasso RMSE for test = ", rsme(y_test, test_set_pred))
print("CATEGORICAL Lasso r2 for test=", r2_score(y_test, test_set_pred))

alphavec = np.linspace(0.01,4,399)
ridge_model = RidgeCV(alphas = alphavec, cv=nfolds)
ridge_model.fit(X_train_scaled, y_train)
print("CATEGORICAL ridge_model.alpha_ = ", ridge_model.alpha_)
val_set_pred = ridge_model.predict(X_val_scaled)
print("CATEGORICAL Ridge RMSE for Val = ", rsme(y_val, val_set_pred))
test_set_pred = ridge_model.predict(X_test_scaled)
print("CATEGORICAL Ridge RMSE for test = ", rsme(y_test, test_set_pred))
print("CATEGORICAL Ridge r2 for test=", r2_score(y_test, test_set_pred))

# ----------------------------------------------------------------------  
#   Find the average alpha values for Lasso and Ridge
# ---------------------------------------------------------------------- 
alpha_Lasso = 0.003 
alpha_Ridge = 3.1578 
has_intercept = True

# Run the corss validation procedure to find the bext model
Run_Kfolds(X_train_val, y_train_val, nfolds, alpha_Lasso, alpha_Ridge, has_intercept)

# Linear Regression yields the highest R-square (best metric among the models)
#   So, now fit the Linear Regression model with the train and validation set 
#   together (last time) to estimate the coefficients. And, test the testing set.
LinearR(X_train_val, y_train_val, X_test, y_test, X, y, features, has_intercept)


