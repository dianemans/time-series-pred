# Import 

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

# FEATURES

def extract(df, name):
    df_name = df[["Date", f'{name}']].copy()
    df_name.set_index('Date', inplace=True)
    df_name.rename(columns={f'{name}': 'return'}, inplace=True)
    df_name.dropna(inplace=True)
    df_name.index = pd.to_datetime(df_name.index)
    return df_name



def Close_price(df, P0 = 100):
    return (1 + df["return"]).cumprod() * P0



def log_return(df):
    return np.log(df['Close'] / df['Close'].shift(1))



def MACD(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return macd



def RSI(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi  



def Drawdown_current(df):
    roll_max = df['Close'].cummax()
    daily_drawdown = df['Close'] / roll_max - 1.0
    return daily_drawdown



def volatility_rolling(df, window):
    return df['log_return'].rolling(window).std()



def feature_engineering_rf(df):
    """
    Applies all transformations and technical indicators to create model features from previous
    functions. This includes MACD, RSI, Drawdown, and rolling volatility.
    Also generates 10 lags of logarithmic returns.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'Close' and 'log_return' columns.

    Returns:
        pd.DataFrame: The enriched DataFrame with all new features (MACD, RSI, etc.), without missing values.
    """
    df['MACD'] = MACD(df)
    df['RSI'] = RSI(df)
    df['Drawdown'] = Drawdown_current(df)
    df['Volatility_20'] = volatility_rolling(df, 20)
    df['Volatility_60'] = volatility_rolling(df, 60)
    lags = list(range(1, 11))
    for lag in lags:
        df[f'lag_{lag}'] = df['log_return'].shift(lag)
    df.dropna(inplace=True)
    return df

def feature_engineering_arp(df, p):
    """
    Applies transformations to create features for AR modeling, including lags of log returns.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'log_return' column.
        p (int): The order of the AR model (number of lags to include).

    Returns:
        pd.DataFrame: The enriched DataFrame with lag features for AR, without missing values.
    """
    lags = list(range(1, p + 1))
    for lag in lags:
        df[f'lag_{lag}'] = df['log_return'].shift(lag)
    df.dropna(inplace=True)
    return df



# WALK FORWARD CROSS VALIDATION 

def WFCV(X, y, model, step_size=50, fold_size=200): 
    """
    Performs Walk-Forward Cross-Validation for time series modeling.
    
    Iteratively trains the model on a rolling window of size `fold_size` 
    and tests it on the subsequent window of size `step_size`.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable (e.g., log returns).
        model (sklearn-like estimator): The machine learning model to be evaluated.
            Must implement `.fit()` and `.predict()` methods.
        step_size (int, optional): The number of steps (days) to predict in each iteration. Defaults to 50.
        fold_size (int, optional): The number of steps (days) used for training in each iteration. Defaults to 200.
        
    Returns:
        tuple: A tuple containing four elements:
            - predictions (np.ndarray): Array of all concatenated predictions.
            - truths (np.ndarray): Array of all concatenated true target values.
            - mse_tab (np.ndarray): Array of the Mean Squared Error (MSE) computed for each fold.
            - r2 (float): The overall R-squared (R2) score across all predictions.
    """

    predictions = []
    truths = []
    mse_tab = []

    for start in range(0, len(X) - fold_size - step_size, step_size):
        train_X = X.iloc[start : start + fold_size] 
        train_y = y.iloc[start : start + fold_size]
        test_X = X.iloc[start + fold_size : start + fold_size + step_size]
        test_y = y.iloc[start + fold_size : start + fold_size + step_size]    

        model.fit(train_X, train_y)
        pred = model.predict(test_X)        
        
        predictions.extend(pred)
        truths.extend(test_y)

        mse = mean_squared_error(truths, predictions)
        mse_tab.append(mse)

    return np.array(predictions), np.array(truths), np.array(mse_tab), r2_score(truths, predictions)

# main 


def stats_forecasting(df_all, name_ticker, model, feature_engineering, plot = True): # plot = True --> show plots and print statements + changer le nom: plus explicite
    """
    Executes the full end-to-end pipeline: data extraction, feature engineering, 
    model training/evaluation (WFCV), and performance analysis.
    
    Args:
        name_ticker (str): The ticker symbol or column name of the asset to process.
        model (sklearn-like estimator): The model to train and evaluate.
        feature_engineering (function): The function to apply for feature engineering on the DataFrame.
        plot (bool, optional): If True, displays progress prints and matplotlib charts. Defaults to True.
        
    Returns:
        dict: A dictionary containing the evaluation metrics of the model:
            - 'name': The processed ticker name.
            - 'MSE': Average Mean Squared Error across all folds.
            - 'OLS_R2': R-squared from the OLS regression (explained variance).
            - 'OLS_Intercept': Intercept of the predictions.
            - 'OLS_Slope': Slope of the predictions.
            - 'OLS_P_Value_Intercept': Statistical significance of the intercept.
            - 'P_Value_Slope': Statistical significance of the slope.
    """

    if plot: print("Processing", name_ticker, '...')

    df = extract(df_all, name_ticker)
    df["Close"] = Close_price(df)
    df["log_return"] = log_return(df)
    df = feature_engineering(df)

    if plot: 
        print("Processing finished")

        print("Data after feature engineering:")
        print(df.head())

        plt.plot(df.index, df['log_return'])
        plt.show()

    X = df.drop(columns=['return', 'log_return', 'Close'])
    y = df['log_return']

    if plot: 
        print("Starting Walk-Forward Cross-Validation...")

    y_pred, y_truth, mse_tab, r2 = WFCV(X, y, model)

    if plot: 
        print("WFCV finished.")

        plt.figure(figsize=(12,6))
        plt.plot(y_truth, label='True Log Returns', color='blue')
        plt.plot(y_pred, label='RF Predicted Log Returns', color='red')
        plt.legend()
        plt.title(f'{name_ticker} Model Predictions vs True Log Returns')
        plt.plot(mse_tab)
        plt.show()

    if plot: 
        print("Computing OLS regression on truth over pred...")

    reg = sm.OLS(y_truth, sm.add_constant(y_pred)).fit()

    if plot: 
        print("Regression complete !")
        print(reg.summary())

    results = {
        'name': name_ticker,
        'MSE': np.mean(mse_tab),
        'OLS_R2': reg.rsquared,          
        'OLS_Intercept': reg.params[0],  
        'OLS_Slope': reg.params[1],     
        'OLS_P_Value_Intercept': reg.pvalues[0],
        'P_Value_Slope': reg.pvalues[1],       
    }

    return results

