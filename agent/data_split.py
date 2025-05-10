from typing import List, Dict, Any
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# ——————————————————————————————
# Implement Leave-One-Out cross-validation on a DataFrame
# using scikit-learn’s LeaveOneOut and a default LinearRegression.

def leave_one_out_cv(df: pd.DataFrame, feature_cols: List[str], target_col: str, model: BaseEstimator = LinearRegression()) -> Dict[str, Any]:
    """
    Perform Leave-One-Out Cross-Validation on the given DataFrame.
    
    Parameters:
        df           : DataFrame containing features and target.
        feature_cols : List of column names to be used as features.
        target_col   : Column name to be used as the target.
        model        : scikit-learn regressor (default: LinearRegression).
        
    Returns:
        dict with 'scores' (array of CV scores) and 'mean_score'.
    """
    
    X = df[feature_cols]
    y = df[target_col]
    
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
    
    return {
        'scores': -scores,
        'mean_score': -np.mean(scores)
    }

# ——————————————————————————————
# Implement a RANSACRegressor on a DataFrame
# returning the fitted model plus inlier/outlier masks.

def ransac_regression(df: pd.DataFrame, feature_cols: List[str], target_col: str, model: BaseEstimator = LinearRegression()) -> Dict[str, Any]:
    """
    Perform RANSAC regression on the given DataFrame.
    
    Parameters:
        df           : DataFrame containing features and target.
        feature_cols : List of column names to be used as features.
        target_col   : Column name to be used as the target.
        model        : scikit-learn regressor (default: LinearRegression).
        
    Returns:
        dict with 'model' (fitted RANSAC model), 'inliers' (mask), and 'outliers' (mask).
    """
    
    X = df[feature_cols]
    y = df[target_col]
    
    ransac = RANSACRegressor(estimator=model)
    ransac.fit(X, y)
    
    inliers = ransac.inlier_mask_
    outliers = np.logical_not(inliers)
    
    return {
        'model': ransac,
        'inliers': inliers,
        'outliers': outliers
    }

def split_data(df: pd.DataFrame, test_size: float = 0.2, shuffle: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Split the provided DataFrame into training and testing sets.
    
    Parameters:
        df        : The DataFrame to split.
        test_size : Proportion of the dataset to allocate to the test set (default 0.2).
        shuffle   : Whether to shuffle the data before splitting (default False).

    Returns:
        A dictionary with keys 'train' and 'test' containing the split DataFrames.
    """

    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=shuffle)
    return {'train': train_df, 'test': test_df}