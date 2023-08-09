from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from AlmaIndicator import ALMAIndicator
import matplotlib.pyplot as plt
from scipy import signal
from numpy import diff
import utility as ut
import pandas as pd
import numpy as np
import tag as t


def plot_data_frame(x_axis, y_axis, colors_point,rsi):
    x_axis = x_axis
    y_axis = y_axis
    colors_point = colors_point
    # plt.scatter(x=x_axis, y=y_axis, c=colors_point)
    list = signal.savgol_filter(y_axis,49,5)
    plt.plot(x_axis, list)
    plt.plot(x_axis, y_axis)
    smooth_peaks, _ = signal.find_peaks(x=list)
    main_peaks, _ = signal.find_peaks(x=y_axis)
    negated_list = [-x for x in y_axis]
    main_valley, _ = signal.find_peaks(x=negated_list)
    np_ax_axis = np.array(x_axis)
    np_ay_axis = np.array(y_axis)
    plt.plot(np_ax_axis[main_peaks], np_ay_axis[main_peaks], 'ro', label='Peaks')
    plt.plot(np_ax_axis[main_valley], np_ay_axis[main_valley], 'go', label='Valleys')
    tag_list = []
    flag = False
    for i in range(len(x_axis)):
        if i in main_valley:
            tag_list.append(True)
            flag = True
        elif i in main_peaks:
            tag_list.append(False)
            flag = False
        else:
            tag_list.append(flag)
    print(tag_list)
    plt.xticks(rotation=90)
    plt.legend()
    #plt.show()
    return tag_list


if __name__ == '__main__':
    df = pd.read_csv(r'/home/araz-abedini-bakhshmand/Documents/ai/random forest/F Historical Data.csv', header=None)
    utility = ut.Utility()
    df = utility.title_label(df)
    df = utility.delete_char(df, 'change')
    df = utility.delete_char(df, 'volume')
    df['close'] = df['close'].astype('float')
    df = df.loc[::-1].reset_index(drop=True)
    price = utility.make_list(df, 'close')
    date_list = utility.make_list(df, 'date')
    alma_indicator = ALMAIndicator(close=df['close'])
    df['alma'] = alma_indicator.alma()
    print(df)
    diff_price = diff(price)
    df = df.iloc[1:]
    df['Price Change'] = diff_price
    period_length = 14
    df['Gain'] = np.where(df['Price Change'] > 0, df['Price Change'], 0)
    df['Loss'] = np.where(df['Price Change'] < 0, abs(df['Price Change']), 0)
    df['Avg Gain'] = df['Gain'].rolling(window=period_length).mean()
    df['Avg Loss'] = df['Loss'].rolling(window=period_length).mean()
    df['RS'] = df['Avg Gain'] / df['Avg Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    change = utility.make_list(df, 'change')
    change = list(map(float, change))
    price = list(map(float, price))
    df['close'] = df['close'].astype('float')
    print(price)
    diff = diff(price)
    tag_list = []
    diff_list = []
    tag_list = utility.tagg(price)
    tag_list = tag_list[1:]
    utility.remove_date(df, utility)
    #df['tag'] = tag_list
    colors = ['green' if tag is True else 'red' for tag in tag_list]
    date_list = date_list[1:]
    price = price[1:]
    rsi = utility.make_list(df,'alma')
    tag_list = plot_data_frame(date_list, price, colors,rsi)
    df['tag'] = tag_list
    df = df.drop(['Avg Gain'], axis=1)
    df = df.drop(['Gain'], axis=1)
    df = df.drop(['Loss'], axis=1)
    df = df.drop(['Avg Loss'], axis=1)
    #df = df.drop(['RSI'], axis=1)
    print(df)
    df = df.iloc[14:]
    X = df.drop(['tag'], axis=1)
    y = df['tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
