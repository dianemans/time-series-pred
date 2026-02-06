# Cache
_cached_data = None

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

def feature_engineering(df):
    df['MACD'] = MACD(df)
    df['RSI'] = RSI(df)
    df['Drawdown'] = Drawdown_current(df)
    df['Volatility_20'] = volatility_rolling(df, 20)
    df['Volatility_60'] = volatility_rolling(df, 60)
    lags = list(range(1, 11))
    for lag in lags:
        df[f'lag_{lag}'] = df['log_return'].shift(lag)
    df.dropna(inplace=True)

# WALK FORWARD CROSS VALIDATION 

def WFCV(X, y, model, step_size=50, fold_size=200): 

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

def get_data():
    global _cached_data
    if _cached_data is None:
        print("--- Loading dataset for the first time ---")
        _cached_data = pd.read_csv('230216_returns.csv')
    return _cached_data

def main(name, model, plot = True): # plot = True --> show plots and print statements

    returns_all = get_data()

    if plot: print("Processing", name, '...')

    df = extract(returns_all, name)
    df["Close"] = Close_price(df)
    df["log_return"] = log_return(df)
    feature_engineering(df)

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
        plt.title(f'{name} Model Predictions vs True Log Returns')
        plt.plot(mse_tab)
        plt.show()

    if plot: 
        print("Computing OLS regression on truth over pred...")

    reg = sm.OLS(y_truth, sm.add_constant(y_pred)).fit()

    if plot: 
        print("Regression complete !")
        print(reg.summary())

    results = {
        'name': name,
        'MSE': np.mean(mse_tab),
        'OLS_R2': reg.rsquared,          
        'OLS_Intercept': reg.params[0],  
        'OLS_Slope': reg.params[1],     
        'OLS_P_Value_Intercept': reg.pvalues[0],
        'P_Value_Slope': reg.pvalues[1],       
    }

    return results

