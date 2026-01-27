# Import 

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# FEATURES

def extract(df, name):
    df_name = df[["Date", f'{name}']].copy()
    df_name.set_index('Date', inplace=True)
    df_name.rename(columns={f'{name}': 'return'}, inplace=True)
    df_name.dropna(inplace=True)
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





