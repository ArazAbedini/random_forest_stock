from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from AlmaIndicator import ALMAIndicator
import matplotlib.pyplot as plt
from numpy import diff
import pandas as pd
import numpy as np
import utility as ut
import tag as t


def k_percent(df):
    df['Lowest Low'] = df['low'].rolling(window=n).min()
    df['Highest High'] = df['high'].rolling(window=n).max()
    df['%K'] = 100 * ((df['close'] - df['Lowest Low']) / (df['Highest High'] - df['Lowest Low']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    df.drop(['Lowest Low', 'Highest High'], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    df = pd.read_csv(r'/home/araz-abedini-bakhshmand/Documents/ai/random forest/F Historical Data.csv', header=None)
    df = df.iloc[1:]
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']
    print(df)
    df.columns = col_names
    utility = ut.Utility()
    df = utility.delete_char(df, 'change')
    df = utility.delete_char(df, 'volume')
    df['close'] = df['close'].astype('float')
    df['Price Change'] = df['close'].diff()
    df = df.iloc[1:]
    period_length = 14
    df['Gain'] = np.where(df['Price Change'] > 0, df['Price Change'], 0)
    df['Loss'] = np.where(df['Price Change'] < 0, abs(df['Price Change']), 0)
    df['Avg Gain'] = df['Gain'].rolling(window=period_length).mean()
    df['Avg Loss'] = df['Loss'].rolling(window=period_length).mean()
    df['RS'] = df['Avg Gain'] / df['Avg Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    n = 14
    df['open'] = df['open'].astype('float')
    df['high'] = df['high'].astype('float')
    df['low'] = df['low'].astype('float')
    df['volume'] = df['volume'].astype('float')
    df['change'] = df['change'].astype('float')
    df['close'] = df['close'].astype('float')
    df = k_percent(df)
    print(df)
