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
import warnings

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# Split train and test sets (indices) from fixed time series data 
from sklearn.model_selection import TimeSeriesSplit


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Calculate RMSE
def rsme(y_true, y_pred):
    '''
    This function calculates the root mean square error
    np.sqrt(np.mean((y_pred - y_true)**2)) 
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


def tune_regularization_hyperparameter(regmodelname, alpha_list, X_train_val, y_train_val, \
                      nfolds, num_entity, panelsubset_index, has_intercept):
    '''
    This function tune the hyperparemeter, alpha parameter in the Lasso and
    Ridge regression models.
    Input arguments:
        regmodelname = model object (Lasso or Ridge)
        alpha_list = Specify a list of alpha values
        X_train_val, y_train_val = train and validation set
        nfolds = number of folds
        num_entity = number of entities, e.g., number of countries
        panelsubset_index = Provide the range of indeces for observations 
                            associated with the entity (or country)        
        has_intercept = a boolean argument. If True, models must include intercept
    '''

    subset_data = X_train_val.iloc[panelsubset_index]
    subset_target = y_train_val.iloc[panelsubset_index]
    subset_time = len(subset_data)
    
    best_alpha = alpha_list[-1]
    best_rmse = 1000
    for alpha in alpha_list:
        
        # Start the cross validation procedure        
        tscv = TimeSeriesSplit(n_splits=10)
        cv_reg_rsme  = []
        for tr_ind, v_ind in tscv.split(subset_data, subset_target):
            
            train_ind = tr_ind.tolist()
            val_ind = v_ind.tolist()
            for ent in range(1, num_entity):
                tr_ind = subset_time + tr_ind
                train_ind += tr_ind.tolist()
                v_ind = subset_time + v_ind
                val_ind += v_ind.tolist()
                
            train_ind = np.array(train_ind)
            val_ind = np.array(val_ind)
            if type(X_train_val) is pd.core.series.Series or type(X_train_val) is pd.core.frame.DataFrame:
                X_train, y_train = X_train_val.iloc[train_ind], y_train_val.iloc[train_ind]
                X_val, y_val = X_train_val.iloc[val_ind], y_train_val.iloc[val_ind]
            else:
                X_train, y_train = X_train_val[train_ind], y_train_val[train_ind]
                X_val, y_val = X_train_val[val_ind], y_train_val[val_ind] 
    
            X_train_scaled = X_train
            X_val_scaled = X_val
            
            # Create models' instance
            if regmodelname == 'Lasso':
                regmodel = Lasso(alpha, fit_intercept=has_intercept)                
            elif regmodelname == 'Ridge':
                regmodel = Ridge(alpha, fit_intercept=has_intercept)
            
            # Fit the model
            regmodel.fit(X_train_scaled, y_train)
            val_yhat = regmodel.predict(X_val_scaled) 
            cv_reg_rsme.append(np.sqrt(mean_squared_error(y_val,val_yhat)))
        
        mean_rmse = np.mean(np.array(cv_reg_rsme))
        if mean_rmse == best_rmse:
            print('best_alpha = ', best_alpha)            
            print('mean_rmse = ', best_rmse)

        if  mean_rmse < best_rmse:   
            best_alpha = alpha
            best_rmse = mean_rmse
    
        
    return regmodel, best_alpha, best_rmse




def run_time_series_KfoldCV(X_train_val, y_train_val, nfolds, alpha_Lasso, alpha_Ridge, \
                            num_entity, panelsubset_index, has_intercept):
    '''
    This function runs the cross validation procedures to find out
        which model returns the best goodness-of-fit metric
    Input arguments:
        X_train_val, y_train_val = train and validation set
        nfolds = number of folds
        alpha_Lasso, alpha_Ridge = the alphas parameters for Lasso and Ridge
                                  respectively
        num_entity = number of entities, e.g., number of countries
        panelsubset_index = Provide the range of indeces for observations 
                            associated with the entity (or country)
        has_intercept = a boolean argument. If True, models must include intercept
    '''        
    # set up containers to collect the metrics that are calculated for 
    #   all models from the validation set 
    cv_lr_r2s, cv_lasso_r2s, cv_ridge_r2s  = [], [], [] 
    cv_lr_Ar2s, cv_lasso_Ar2s, cv_ridge_Ar2s  = [], [], [] 
    cv_lr_rsme, cv_lasso_rsme, cv_ridge_rsme  = [], [], [] 
    
    # Start the cross validation procedure        
    tscv = TimeSeriesSplit(n_splits=10)

    subset_data = X_train_val.iloc[panelsubset_index]
    subset_target = y_train_val.iloc[panelsubset_index]
    subset_time = len(subset_data)
    
    for tr_ind, v_ind in tscv.split(subset_data, subset_target):
        
        train_ind = tr_ind.tolist()
        val_ind = v_ind.tolist()
        for ent in range(1, num_entity):
            tr_ind = subset_time + tr_ind
            train_ind += tr_ind.tolist()
            v_ind = subset_time + v_ind
            val_ind += v_ind.tolist()
            
        train_ind = np.array(train_ind)
        val_ind = np.array(val_ind)
        if type(X_train_val) is pd.core.series.Series or type(X_train_val) is pd.core.frame.DataFrame:
            X_train, y_train = X_train_val.iloc[train_ind], y_train_val.iloc[train_ind]
            X_val, y_val = X_train_val.iloc[val_ind], y_train_val.iloc[val_ind]
        else:
            X_train, y_train = X_train_val[train_ind], y_train_val[train_ind]
            X_val, y_val = X_train_val[val_ind], y_train_val[val_ind] 

        X_train_scaled = X_train
        X_val_scaled = X_val
        
        '''
        # IGNORE: Apply standardScaler to training set
        std = StandardScaler()
        std.fit(X_train.values)    
        X_train_scaled = std.transform(X_train.values)
        #Apply the scaler to the val and test set
        X_val_scaled = std.transform(X_val.values)
        '''
        
        # Create models' instance
        lr_model = LinearRegression(fit_intercept=has_intercept)
        lasso_model = Lasso(alpha_Lasso, fit_intercept=has_intercept)
        ridge_model = Ridge(alpha_Ridge, fit_intercept=has_intercept)
        
        # Fit the model
        lasso_model.fit(X_train_scaled, y_train)
        ridge_model.fit(X_train_scaled, y_train)    
        lr_model.fit(X_train_scaled, y_train)   
        
        # Store r2
        cv_lr_r2s.append(lr_model.score(X_val_scaled, y_val))
        cv_lasso_r2s.append(lasso_model.score(X_val_scaled, y_val))
        cv_ridge_r2s.append(ridge_model.score(X_val_scaled, y_val))
        
        # Store Adjusted r2
        cv_lr_Ar2s.append(AdjR2s(X_val_scaled, y_val, cv_lr_r2s[-1]))
        cv_lasso_Ar2s.append(AdjR2s(X_val_scaled, y_val, cv_lasso_r2s[-1]))
        cv_ridge_Ar2s.append(AdjR2s(X_val_scaled, y_val, cv_ridge_r2s[-1]))
        
        # Store RSME
        val_yhat = lr_model.predict(X_val_scaled)
        cv_lr_rsme.append(np.sqrt(mean_squared_error(y_val,val_yhat)))
        
        val_yhat = lasso_model.predict(X_val_scaled) 
        cv_lasso_rsme.append(np.sqrt(mean_squared_error(y_val,val_yhat)))
        
        val_yhat = ridge_model.predict(X_val_scaled)
        cv_ridge_rsme.append(np.sqrt(mean_squared_error(y_val,val_yhat)))
        
    # Ignore verbose: print the all results
    #print('\nLinear Regression R^2: ', cv_lr_r2s)
    #print('\nLasso R^2: ', cv_lasso_r2s)
    #print('\nRidge R^2: ', cv_ridge_r2s)

    #print('\nLinear Regression: Adjusted R^2: ', cv_lr_Ar2s)
    #print('\nLasso Adjusted R^2: ', cv_lasso_Ar2s)
    #print('\nRidge Adjusted R^2: ', cv_ridge_Ar2s)
    
    #print('\nLinear Regression: RMSE: ', cv_lr_rsme)
    #print('\nLasso RSME: ', cv_lasso_rsme)
    #print('\nRidge RSME: ', cv_ridge_rsme)
    
    print(f'\nLinear Regression Average R^2: {np.mean(cv_lr_r2s):.3f}, std = {np.std(cv_lr_r2s):.3f}')
    print(f'Lasso mean Average R^2: {np.mean(cv_lasso_r2s):.3f}, std = {np.std(cv_lasso_r2s):.3f}')    
    print(f'Ridge mean Average R^2: {np.mean(cv_ridge_r2s):.3f}, std = {np.std(cv_ridge_r2s):.3f}')    

    print(f'\nLinear Regression Average RMSE: {np.mean(cv_lr_rsme):.3f}, std = {np.std(cv_lr_rsme):.3f}')
    print(f'Lasso mean Average RMSE: {np.mean(cv_lasso_rsme):.3f}, std = {np.std(cv_lasso_rsme):.3f}')    
    print(f'Ridge mean Average RMSE: {np.mean(cv_ridge_rsme):.3f}, std = {np.std(cv_ridge_rsme):.3f}')    
     
    return print('\nCross validation has completed.')
    


def LinearR(X_train_val, y_train_val, X_test, y_test, features, has_intercept):
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
     - print the regression's coeeficients
    '''    
    
    
    '''
    # IGNORE: Apply Standard Scaler to training set
    std = StandardScaler()
    std.fit(X_train_val.values)    
    X_train_val_scaled = std.transform(X_train_val.values)
    #Apply the scaler to the test set
    X_test_scaled = std.transform(X_test.values)
    '''
    
    X_train_val_scaled = X_train_val
    X_test_scaled = X_test
    
    # # fit linear regression to train+val data
    lr_model = LinearRegression(fit_intercept=has_intercept)
    lr_model.fit(X_train_val_scaled, y_train_val)
    
    # score fit model on test data
    test_rsquare = lr_model.score(X_test_scaled, y_test)
    test_adj_rsquare = 1 - (1-test_rsquare)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    test_yhat = lr_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test,test_yhat))
    
    # Report R-square, adjusted R-square, and RSME for the test data 
    print('Linear Regression: R^2 score: ', test_rsquare)
    print('Linear Regression: Adjusted R^2 score: ', test_adj_rsquare)
    print('Linear Regression: RMSE: ', test_rmse)
    
    # Plot: predicted target vs actual target for the test set
    plt.scatter(test_yhat, y_test, color='g')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('y', size=12)
    plt.title('Linear Regression: Predicted Vs Actual for Test data')
    plt.show()
    
    # Plot: residuals vs predicted target for the test set
    res = y_test.values- test_yhat
    plt.scatter(test_yhat, res, color='g')
    plt.axhline(0,color='r', linestyle='--')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('Residuals', size=12)
    plt.title('Linear Regression: Predicted Vs Residuals for Testdata')   
    plt.show()
    
    if has_intercept:
        print('intercept', lr_model.intercept_)
    #Plot: a bar graph to visualize the regression coeeficients  
    fea_len = len(features)
    
    if 'code' in features:
        finalfea_df = pd.DataFrame(lr_model.coef_[0:fea_len-1],index=features[1:])
        print('\n The coefficients')
        print(finalfea_df)
        #coefs = pd.Series(lr_model.coef_[0:fea_len-1], features[1:]).sort_values()
    
    #coefs.plot(kind='bar', title="Linear Regression's Coefficients")    
    #plt.show()
    
    return lr_model
    
def Regularized_regression_model(X_train_val, y_train_val, X_test, y_test, features, alpha_Lasso, alpha_Ridge, has_intercept):
    '''
    This function fits the LASSO and RIDGE models with training and validation
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
     - print the regression's coeeficients
    '''    
    
    '''
    # IGNORE: Apply Standard Scaler to training set
    std = StandardScaler()
    std.fit(X_train_val.values)    
    X_train_val_scaled = std.transform(X_train_val.values)
    #Apply the scaler to the test set
    X_test_scaled = std.transform(X_test.values)
    '''
    X_train_val_scaled = X_train_val
    X_test_scaled = X_test
    
    # # fit the models to train+val data
    lasso_model = Lasso(alpha = alpha_Lasso, fit_intercept=has_intercept)
    ridge_model = Ridge(alpha = alpha_Ridge, fit_intercept=has_intercept) #0.3
    
    lasso_model.fit(X_train_val_scaled, y_train_val)
    ridge_model.fit(X_train_val_scaled, y_train_val)

    # ************************************* 
    # *************** LASSO ***************
    # *************************************     
    # LASSO: core fit model on test data
    test_rsquare = lasso_model.score(X_test_scaled, y_test)
    test_adj_rsquare = 1 - (1-test_rsquare)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    test_yhat = lasso_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test,test_yhat))
    
    # Report R-square, adjusted R-square, and RSME for the test data 
    print('Lasso: R^2 score: ', test_rsquare)
    print('Lasso: Adjusted R^2 score: ', test_adj_rsquare)
    print('Lasso: RMSE: ', test_rmse)
    
    
    # Plot: predicted target vs actual target for the test set
    plt.scatter(test_yhat, y_test, color='b')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('y', size=12)
    plt.title('Lasso: Predicted Vs Actual for TEST data')
    plt.show()
    
    # Plot: residuals vs predicted target for the test set
    res = y_test- test_yhat
    plt.scatter(test_yhat, res, color='b')
    plt.axhline(0,color='r', linestyle='--')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('Residuals', size=12)
    plt.title('Lasso: Predicted Vs Residuals for TEST data')   
    plt.show()

    
    if has_intercept:
        print('intercept', lasso_model.intercept_)
    
    #Plot: a bar graph to visualize the regression coeeficients  
    fea_len = len(features)
    if 'code' in features:
        finalfea_df = pd.DataFrame(lasso_model.coef_[0:fea_len-1],index=features[1:])
        print('\n The coefficients')
        print(finalfea_df)        
        #coefs = pd.Series(lasso_model.coef_[0:fea_len-1], features[1:]).sort_values()
        
    #coefs.plot(kind='bar', title="Lasso's Coefficients")    
    #plt.show()

    # ************************************* 
    # *************** RIDGE ***************
    # *************************************     
    # RIDGE: core fit model on test data
    test_rsquare = ridge_model.score(X_test_scaled, y_test)
    test_adj_rsquare = 1 - (1-test_rsquare)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    test_yhat = ridge_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test,test_yhat))
    
    # Report R-square, adjusted R-square, and RSME for the test data 
    print('Ridge: R^2 score: ', test_rsquare)
    print('Ridge: Adjusted R^2 score: ', test_adj_rsquare)
    print('Ridge: RMSE: ', test_rmse)
    
    
    # Plot: predicted target vs actual target for the test set
    plt.scatter(test_yhat, y_test, color='m')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('y', size=12)
    plt.title('Ridge: Predicted Vs Actual for TEST data')
    plt.show()
    
    # Plot: residuals vs predicted target for the test set
    res = y_test- test_yhat
    plt.scatter(test_yhat, res, color='m')
    plt.axhline(0,color='r', linestyle='--')
    plt.xlabel('y_hat', size=12)
    plt.ylabel('Residuals', size=12)
    plt.title('Ridge: Predicted Vs Residuals for TEST data')   
    plt.show()

    
    if has_intercept:
        print('intercept', ridge_model.intercept_)
    #Plot: a bar graph to visualize the regression coeeficients  
    fea_len = len(features)
    
    if 'code' in features:
        finalfea_df = pd.DataFrame(ridge_model.coef_[0:fea_len-1],index=features[1:])
        print('\n The coefficients')
        print(finalfea_df)
        #coefs = pd.Series(ridge_model.coef_[0:fea_len-1], features[1:]).sort_values()
    
    #coefs.plot(kind='bar', title="Ridge's Coefficients")    
    #plt.show()

    return lasso_model, ridge_model


def plot_yield_selected_country(model, country_code, year_train_range, year_test_range, \
                                country_title, X_train_val, y_train_val, \
                                X_test, y_test):
    '''
    This function generate a year Vs cereal yield plot with 
    markers represent the actual data and dotted lines represent
    the predicted cereal yield.
    
    Input arguments:
    model = a specific regression model
    country_code = a country's code that are encoded as a dummy variable
    year_train_range = a range of years associated to the training dataset
    year_test_range = a range of years associated to the test dataset
    country_title = specify the country code
    X_train_val, y_train_val = training and validation sets
    X_test, y_test = test sets
    
    Output:
        year Vs cereal yield plot
    '''
    X_trainval_subset = X_train_val[X_train_val[country_code]==1]
    idx = X_trainval_subset.index.values
    y_trainval_subset = y_train_val.loc[idx]
    
    X_test_subset = X_test[X_test[country_code]==1]
    idx = X_test_subset.index.values
    y_test_subset = y_test.loc[idx]
    
    y_trainval_hat = model.predict(X_trainval_subset)
    y_test_hat = model.predict(X_test_subset)
    
    plt.scatter(year_train_range, y_trainval_subset, c='k')
    plt.scatter(year_test_range, y_test_subset, c='b')
    
    plt.plot(year_train_range, y_trainval_hat, c='m', linestyle='-.')
    plt.plot(year_test_range, y_test_hat, c='r', linestyle='-.')
    plt.xlabel('Year')
    plt.ylabel('Cereal Yield')
    plt.title(country_title)
    plt.show()
    


if __name__ == "__main__":
    
    #********** Training and Testing the models ****************
    # Load the pickled DataFrames from the data/processed folder
    df = pd.read_pickle('df_train_clean.pickle')
    df_test = pd.read_pickle('df_test_clean.pickle')
    
    # ----------------------------------------------------------------------------
    # Fixed effects regresion model:  
    #     OLS regression model called least square with dummy variables model
    # ----------------------------------------------------------------------------
    # perform train/val/test split for time series panel data
    features = ['code', 'A', 'V', 'E', 'F', 'M', 'Gc', 'Dl', 'R', 'Tavg', 'Rmin']
    df_X = df[(df['year']>='1991') & (df['year']<='2005')]
    df_X_val = df[(df['year']>='2006') & (df['year']<='2010')]
        
    # Extract the target, y
    y_train = df_X['Ckg'] 
    y_val = df_X_val['Ckg']
    y_test = df_test['Ckg']
    
    print('Generate dummy variables for the entities (countries)')
    df_X = df_X.loc[:,features]
    df_X_val = df_X_val.loc[:,features]
    df_X_test = df_test.loc[:,features]
    
    # Generate dummy variables for the entities (countries)
    X_train = pd.get_dummies(df_X, prefix='code', columns=['code'])
    X_val = pd.get_dummies(df_X_val, prefix='code', columns=['code'])
    X_test = pd.get_dummies(df_X_test, prefix='code', columns=['code'])
    
    X_train_val = df.loc[:,features]
    y_train_val = df['Ckg'] 
    X_train_val = pd.get_dummies(X_train_val, prefix='code', columns=['code'])
    panelsubset_index = [i for i in range(0,20)]
    
    
    
    # Ignore this step: Apply Standard Scaler to training set
    '''
    std = StandardScaler()
    std.fit(X_train.values)    
    X_train_scaled = std.transform(X_train.values)
    #Apply the scaler to the val and test set
    X_val_scaled = std.transform(X_val.values)
    X_test_scaled = std.transform(X_test.values)
    '''

    # ----------------------------------------------------------------------  
    #   Find the optimum alpha parameters for Lasso and Ridge
    # ----------------------------------------------------------------------
    '''
    Results:
    
    LASSO: LEAST SQUARE WITH DUMMY VARIABLES MODEL
    LSDV: Lasso alpha_ =  0.001
    LSDV: Lasso RMSE for Val =  845.0933278731812
    LSDV: Lasso RMSE for test =  1349.6455959012378
    LSDV: Lasso r2 for test= 0.7501607919844578
    RIDGE: LEAST SQUARE WITH DUMMY VARIABLES MODEL
    LSDV: Ridge alpha_ =  0.01
    LSDV: Ridge RMSE for Val =  861.7397637878266
    LSDV: Ridge RMSE for test =  1345.0041904586321
    LSDV: Ridge r2 for test= 0.7518762217736438       
    '''
    
    nfolds = 10
    num_entity = 111   
    has_intercept = True    
    
    print("LASSO: LEAST SQUARE WITH DUMMY VARIABLES MODEL")  
    alphavec = np.linspace(0.001,2,99)
    lasso_model, bestalpha, best_rmse = tune_regularization_hyperparameter('Lasso', alphavec, X_train_val, y_train_val, \
                      nfolds, num_entity, panelsubset_index, has_intercept)
    print("LSDV: Lasso alpha_ = ", bestalpha)
    val_set_pred = lasso_model.predict(X_val)
    print("LSDV: Lasso RMSE for Val = ", np.sqrt(mean_squared_error(y_val, val_set_pred)))
    test_set_pred = lasso_model.predict(X_test)
    print("LSDV: Lasso RMSE for test = ", np.sqrt(mean_squared_error(y_test, test_set_pred)))
    print("LSDV: Lasso r2 for test=", r2_score(y_test, test_set_pred))    

    
    print("RIDGE: LEAST SQUARE WITH DUMMY VARIABLES MODEL")
    alphavec = np.linspace(0.01,2, 99)
    ridge_model, bestalpha, best_rmse = tune_regularization_hyperparameter('Ridge', alphavec, X_train_val, y_train_val, \
                      nfolds, num_entity, panelsubset_index, has_intercept)
    print("LSDV: Ridge alpha_ = ", bestalpha)
    val_set_pred = ridge_model.predict(X_val)
    print("LSDV: Ridge RMSE for Val = ", np.sqrt(mean_squared_error(y_val, val_set_pred)))
    test_set_pred = ridge_model.predict(X_test)
    print("LSDV: Ridge RMSE for test = ", np.sqrt(mean_squared_error(y_test, test_set_pred)))
    print("LSDV: Ridge r2 for test=", r2_score(y_test, test_set_pred)) 
    
    
    # ----------------------------------------------------------------------  
    #   Find the average alpha values for Lasso and Ridge
    # ---------------------------------------------------------------------- 
    alpha_Lasso = 0.001
    alpha_Ridge = 0.01
    has_intercept = True
    nfolds = 10
    num_entity = 111
    
    ## Run the corss validation procedure to find the bext model
    #run_time_series_KfoldCV(X_train_val, y_train_val, nfolds, alpha_Lasso, 
    #    alpha_Ridge, num_entity, panelsubset_index, has_intercept)
    
    ## Linear Regression yields the highest R-square and RMSE (best metric among the models)
    ##   So, now fit the Linear Regression model with the train and validation set 
    ##   together (last time) to estimate the coefficients. And, test the testing set.
    lrmodel = LinearR(X_train_val, y_train_val, X_test, y_test,features, has_intercept)
    #lasso_model, ridge_model = Regularized_regression_model(X_train_val, y_train_val, X_test, y_test, 
    #                             features, alpha_Lasso, alpha_Ridge, has_intercept)
    
    # -----------------------------------------------------------------
    #    Plot specific country
    # -----------------------------------------------------------------
    country_code = 'code_SEN'#'code_GHA' #'code_IDN' #'code_IND' #'code_CAN' #'code_MYS' #'code_BRA'  #'code_USA'
    country_title = 'Senegal'#'Ghana' #'Indonesia' #'India' #'Canada'
    year1 = range(1991,2011)
    year2 = range(2011,2016)
    
    plot_yield_selected_country(lrmodel, country_code, year1, year2, \
                                country_title, X_train_val, y_train_val, \
                                X_test, y_test)
    
    
