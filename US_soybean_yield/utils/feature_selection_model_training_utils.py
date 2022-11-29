import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost
SEED = 42

def variance_threshold_selector(df, threshold):
    """

    parameters
    ----------
    df: dataframe
        dataframe that contains the feature information
        in our case, df will be either v0_X_train_df or v1_X_train_df

    threshold: float
        threshold used to eliminate low variance variables

    returns
    ----------
    two dataframes:
    dataframe with high variance, and dataframe with low variance

    """

    selector = VarianceThreshold(threshold)

    # Note: here we call selector.fit, not selector.fit_transform, as the latter will directly return an array
    #       with low variance features dropped

    selector.fit(df)

    mask     = selector.get_support()

    df_high_variance = df[df.columns[mask]]

    df_low_variance  = df[df.columns[~mask]]

    return df_high_variance, df_low_variance

def f_regression_selector(X_train, y_train, feature_columns, k_value):
    """
    Function to do feature selection from linear regression using the F-statistics
    Adopted from
    https://www.featureranking.com/tutorials/machine-learning-tutorials/sk-part-2-feature-selection-and-ranking/
    https://www.rasgoml.com/feature-engineering-tutorials/feature-selection-using-the-f-test-in-scikit-learn

    parameters
    ----------
    X_train: DataFrame
        Features from the training dataset

    y_train: DataFrame
        Target from the training dataset

    X_test: DataFrame
        Features from the training dataset

    feature_columns: List of strings
        Column names of featrures in X_train and X_test

    k_value: Number of top features to select.
        The “all” option bypasses selection, for use in a parameter search.

    returns
    ----------

    """

    # configure to select k features
    fs    = SelectKBest(score_func=f_regression, k=k_value)

    # obtain numeric values from the dataframes
    X_train_values   = X_train.values
    y_train_values   = y_train.values

    # scaliing
    scaler = StandardScaler()
    X_train_scaled   = scaler.fit_transform(X_train_values)

    # get the top k_value features
    X_train_selected = fs.fit_transform(X_train_scaled, y_train_values)

    if type(k_value) == int:

        # get the feature indices
        fs_indices = np.argsort(np.nan_to_num(fs.scores_))[::-1][0:k_value]

    else:
        # get the feature indices
        fs_indices = np.argsort(np.nan_to_num(fs.scores_))[::-1]

    # get the column names
    fs_names   = feature_columns[fs_indices].values

    # get teh fscore for the selected columns
    fs_fscore  = fs.scores_[fs_indices]

    # return fs_fscore, fs_names
    return fs_fscore, fs_names

def mutual_info_regression_selector(X_train, y_train, feature_columns, k_value):
    """
    Function to do feature selection from linear regression using the F-statistics
    Adopted from
    https://www.featureranking.com/tutorials/machine-learning-tutorials/sk-part-2-feature-selection-and-ranking/
    https://www.rasgoml.com/feature-engineering-tutorials/feature-selection-using-the-f-test-in-scikit-learn

    parameters
    ----------
    X_train: DataFrame
        Features from the training dataset

    y_train: DataFrame
        Target from the training dataset

    X_test: DataFrame
        Features from the training dataset

    feature_columns: List of strings
        Column names of featrures in X_train and X_test

    k_value: Number of top features to select.
        The “all” option bypasses selection, for use in a parameter search.

    returns
    ----------

    """

    # configure to select k features
    fs    = SelectKBest(score_func=mutual_info_regression, k=k_value)

    # obtain numeric values from the dataframes
    X_train_values   = X_train.values
    y_train_values   = y_train.values

    # scaliing
    scaler = StandardScaler()
    X_train_scaled   = scaler.fit_transform(X_train_values)

    # get the top k_value features
    X_train_selected = fs.fit_transform(X_train_scaled, y_train_values)

    if type(k_value) == int:

        # get the feature indices
        fs_indices = np.argsort(np.nan_to_num(fs.scores_))[::-1][0:k_value]

    else:
        # get the feature indices
        fs_indices = np.argsort(np.nan_to_num(fs.scores_))[::-1]

    # get the column names
    fs_mutual_names = feature_columns[fs_indices].values

    # get teh fscore for the selected columns
    fs_mutual_info  = fs.scores_[fs_indices]

    # return fs_fscore, fs_names
    return fs_mutual_info, fs_mutual_names

def plot_importance(fs_names, fs_fscore, method_name, df_name):
    """
    https://www.featureranking.com/tutorials/machine-learning-tutorials/sk-part-2-feature-selection-and-ranking/

    parameters
    ----------
    fs_names: array of strings
        feature names

    fs_fscore: array of numerical values
        fscore of selected features

    method_name: string
        method used for feature selection

    df_name: string
        dataframe used for feature selection

    returns
    ----------
    Bar plot showing the feature importance of selected features.

    """
    plt.style.use("ggplot")
    plt.barh(fs_names, fs_fscore)
    plt.title(df_name+' : '+method_name + ' Feature Importances')
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

def lasso_selector(X_train, y_train, alpha):
    """
    Feature selection using Lasso regression.
    
    Parameters
    ----------
    X_train: dataframe
    
    y_train: dataframe
    
    alpha: float
    
    Returns
    ----------
    Names of important features
    
    Plot of feature importance
    
    """
    
    # define a lasso pipeline
    lasso_pipeline = Pipeline([
                                ('scaler', StandardScaler()), \
                                ('lasso', Lasso(alpha=alpha, random_state=SEED))    
                             ])
    
    # obtain numeric values in the dataframe
    X_train_values = X_train.values
    y_train_values = y_train.values
    
    # fit a model
    lasso = lasso_pipeline.fit(X_train_values, y_train_values)
    
    # obtain the coefficients 
    lasso_coef = lasso.named_steps['lasso'].coef_
    
    # obtain the importance 
    lasso_importance = np.abs(lasso_coef)
    
    # obtain the name of important features
    all_features     = X_train.columns
    
    lasso_features   = all_features[lasso_importance > 0]
    
    # obtain the coef of important features
    lasso_features_coef = lasso_importance[lasso_importance > 0]
    
    # return features and coef
    return lasso_features, lasso_features_coef

def sel_decisiontree(X_train, y_train):
    """
    Feature selection by applying SelectFromModel method to DecisionTreeRegressor

    Parameters
    ----------
    X_train: dataframe

    y_train: dataframe

    Returns
    ----------
    X_train_drop
    X_train_keep

    """

    # obtain the numeric values
    X_train_values = X_train.values
    y_train_values = y_train.values

    # scaling
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_values)

    # call SelectFromModel
    sel = SelectFromModel(DecisionTreeRegressor(random_state=SEED)).fit(X_train_scaled, y_train_values)

    # get the mask of selected features
    sel_mask = sel.get_support()

    #sel_features = X_train.columns[(sel.get_support())]

    #sel_importance = sel._feature_importances_

    X_train_keep = X_train[X_train.columns[sel_mask]]

    X_train_drop = X_train[X_train.columns[~sel_mask]]

    return X_train_drop, X_train_keep

def sel_randomforest(X_train, y_train):
    """
    Feature selection by applying SelectFromModel method to RandomForestRegressor

    Parameters
    ----------
    X_train: dataframe

    y_train: dataframe

    Returns
    ----------
    X_train_drop
    X_train_keep

    """

    # obtain the numeric values
    X_train_values = X_train.values
    y_train_values = y_train.values

    # scaling
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_values)

    # call SelectFromModel
    sel = SelectFromModel(RandomForestRegressor(random_state=SEED)).fit(X_train_scaled, y_train_values)

    # get the mask of selected features
    sel_mask = sel.get_support()

    #sel_features = X_train.columns[(sel.get_support())]

    #sel_importance = sel._feature_importances_

    X_train_keep = X_train[X_train.columns[sel_mask]]

    X_train_drop = X_train[X_train.columns[~sel_mask]]

    return X_train_drop, X_train_keep

def sel_xgboost(X_train, y_train):
    """
    Feature selection by applying SelectFromModel method to xgboost.XGBRegressor

    Parameters
    ----------
    X_train: dataframe

    y_train: dataframe

    Returns
    ----------
    X_train_drop
    X_train_keep

    """

    # obtain the numeric values
    X_train_values = X_train.values
    y_train_values = y_train.values

    # scaling
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_values)

    # call SelectFromModel
    sel = SelectFromModel(xgboost.XGBRegressor(random_state=SEED)).fit(X_train_scaled, y_train_values)

    # get the mask of selected features
    sel_mask = sel.get_support()

    #sel_features = X_train.columns[(sel.get_support())]

    #sel_importance = sel._feature_importances_

    X_train_keep = X_train[X_train.columns[sel_mask]]

    X_train_drop = X_train[X_train.columns[~sel_mask]]

    return X_train_drop, X_train_keep

