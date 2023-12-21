from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

import utility as ut
import win_rate_test
import pandas as pd
import numpy as np



def extremum_list(y_axis: list) -> np.array:

    negated_list = [-x for x in y_axis]
    peaks, _ = signal.find_peaks(x=y_axis)
    valleys, _ = signal.find_peaks(x=negated_list)
    arr = np.empty((peaks.size + valleys.size), dtype='i4')
    print(len(peaks))
    print(len(valleys))
    print(len(arr))
    arr[0::2] = valleys[:]
    arr[1::2] = peaks[:]
    return arr

def find_big_difference(y_axis: list) -> tuple:
    arr = extremum_list(y_axis)
    max_inc = np.diff(y_axis[arr]).max()
    max_inc_index = np.argmax(max_inc)
    max_dec = np.diff(y_axis[arr]).min()
    max_dec_index = np.argmin(max_dec)
    inc = max_inc / arr[max_inc_index + 1]
    dec = max_dec / arr[max_dec_index + 1]
    tuple = (inc, dec)
    return tuple

def find_small_difference(y_axis: list) -> tuple:
    arr = extremum_list(y_axis)
    diff_array = np.diff(y_axis[arr])
    min_value = 0
    min_index = None
    max_value = -10000000
    max_index = None
    for i in range(len(diff_array)):
        if diff_array[i] >= 0 and diff_array[i] > min_value:
            min_index = i
            min_value = diff_array[i]
        if diff_array[i] <= 0 and diff_array[i] > max_value:
            max_index = i
            max_value = diff_array[i]
    if max_index is not None:
        inc = min_value / arr[min_index + 1]
    else:
        inc = 0
    if min_index is not None:
        dec = max_value / arr[max_index + 1]
    else:
        dec = 0
    tuple = (inc, dec)
    return tuple

def cut_signal(y_axis: list, CONSTANT=0.95):
    y_field = np.array(y_axis)
    take_profit = y_field * (1 - CONSTANT) + y_field
    stopless = y_field - y_field * (1 - CONSTANT)
    return take_profit, stopless


def mean_calculate(y_axis: list):
    arr = extremum_list(y_axis)
    diff_array = np.diff(y_axis[arr])

    price = y_axis[arr][:-1]
    mean_list = diff_array / price
    inc_mean = mean_list[mean_list >= 0].mean()
    dec_mean = mean_list[mean_list < 0].mean()
    return inc_mean, dec_mean


def regression_model(y_axis: list):
    arr = extremum_list(y_axis)
    diff_array = np.diff(y_axis[arr])
    y = np.array(y_axis[arr])
    valleys = y[0::2]
    peaks = y[1::2]
    X = np.linspace(0, len(y), len(y))
    x_peaks = np.arange(len(peaks)).reshape((-1, 1))
    x_valleys = np.arange(len(valleys)).reshape((-1, 1))
    model_peaks = make_pipeline(PolynomialFeatures(17), LinearRegression())
    model_peaks.fit(x_peaks, peaks)
    y_peaks = model_peaks.predict(np.array(len(peaks)).reshape(-1, 1))
    model_valleys = make_pipeline(PolynomialFeatures(17), LinearRegression())
    model_valleys.fit(x_valleys, valleys)
    y_valleys = model_peaks.predict(np.array(len(peaks)).reshape(-1, 1))
    return y_peaks, y_valleys

if __name__ == '__main__':
    df = pd.read_json(r'/home/araz/Documents/ai/files/XAUUSD_candlestick_1D.json',convert_dates=True)
    train_length = int(len(df) * 0.8)
    df_test = df[train_length - 15:]
    df_test.to_json('data.json', orient='records')
    utility = ut.Utility()
    df = df.dropna()
    price = utility.make_list(df, 'Close')
    df = utility.alma_calculator(data_frame=df)
    df = df[1:]
    df = utility.calculate_rsi(df, price)
    tag_list = []
    diff_list = []
    df.dropna(inplace=True)
    date_list = utility.make_list(df, 'Open Time')
    df.drop('Open Time', inplace=True, axis=1)
    date_list = np.array(list(map(str, date_list)))
    price = utility.make_list(df, 'Close')
    inc, dec = find_big_difference(y_axis=price)
    print(inc, '         ', dec)
    inc, dec = find_small_difference(y_axis=price)
    print(inc, '         ', dec)
    cut_signal(y_axis=price)
    tag_list = ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']
    regression_model(y_axis=price)