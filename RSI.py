from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from AlmaIndicator import ALMAIndicator
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import signal
from numpy import diff
import seaborn as sns
import risk_to_reward
import win_rate as wr
import utility as ut
import win_rate_test
import pandas as pd
import numpy as np
import xlsxwriter
import tag as t
import pickle
import limit



def ploting(date_list,price, tag_list):
    colors = []
    for index in tag_list:
        if index == 'B':
            colors.append('green')
        elif index == 'S':
            colors.append('red')
        elif index == 'S/B':
            colors.append('orange')
        else:
            colors.append('black')
    plt.plot(date_list, price)
    plt.scatter(date_list, price, c=colors)
    plt.xticks(rotation=90)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('total labeling')
    plt.show()
def top_max_peaks(smooth_peaks, y_axis):
    start_point = 0
    max_points = []
    for index in smooth_peaks:
        current_list = y_axis[start_point:index + 1]
        max_point = max(current_list)
        max_index = current_list.index(max_point) + start_point
        max_points.append(max_index)
        start_point = index

    np_max = np.array(max_points)
    return np_max


def top_min_peaks(smooth_peaks, y_axis):
    start_point = 0
    min_points = []
    for index in smooth_peaks:
        current_list = y_axis[start_point:index + 1]
        min_point = min(current_list)
        min_index = current_list.index(min_point) + start_point
        min_points.append(min_index)
        start_point = index
    np_min = np.array(min_points)
    return np_min






def plot_data_frame(x_axis, y_axis):
    x_axis = x_axis.tolist()
    y_axis = y_axis.tolist()
    list = signal.savgol_filter(y_axis,21, 3)
    smooth_peaks, _ = signal.find_peaks(x=list)
    main_peaks, _ = signal.find_peaks(x=y_axis)
    negated_list = [-x for x in y_axis]
    negated_smooth_list = [-x for x in list]
    main_valley, _ = signal.find_peaks(x=negated_list)
    min_smooth_peaks, _ = signal.find_peaks(x=negated_smooth_list)
    max_points = top_max_peaks(smooth_peaks, y_axis)
    min_points = top_min_peaks(min_smooth_peaks, y_axis)
    tag_list = []
    red_list = []
    green_list = []
    flag = True
    signal_list = []
    buy_price = None
    buy = None
    row = 0
    just_buy = 0
    just_sell = 0
    for i in range(len(x_axis)):
        if i in min_points:
            flag = True
            red_list.append(i)
            tag_list.append('B')
            signal_list.append('B')
            buy = x_axis[i]
            buy_price = y_axis[i]
            just_buy += 1
        elif i in max_points:
            green_list.append(i)
            signal_list.append('S')
            tag_list.append('S')
            just_sell += 1
            if buy != None:
                row += 1
            flag = False

        else:
            if flag == True:
                red_list.append(i)
                try:
                    risk_reward = (0.95 * buy_price) / (y_axis[i] - buy_price)
                    if risk_reward < 1.5:
                        signal_list.append('B')
                        tag_list.append('B')
                        just_buy += 1
                    else:
                        signal_list.append('B')
                        tag_list.append('B')
                        buy_price = y_axis[i]
                        row += 1
                        buy = x_axis[i]
                        just_buy += 1
                except:
                    signal_list.append('B')
                    tag_list.append('B')
                    just_buy += 1
            else:
                green_list.append(i)
                signal_list.append('S')
                tag_list.append('S')
                just_sell += 1
    ploting(x_axis, y_axis, tag_list)
    print('just buy : ', just_buy)
    print('just sell : ', just_sell)
    return tag_list, signal_list

def write_model(X_train, y_train):
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    with open('/home/araz/Documents/ai/stock/model/daily.obj', 'wb') as f:
      pickle.dump(rfc, f)

def read_model():
    with open('/home/araz/Documents/ai/stock/model/daily.obj', 'rb') as f:
        rfc = pickle.load(f)
    return rfc
def train_test(df):
    X = df.drop(['tag', 'id', 'RS', 'High', 'Low', 'Price Change'], axis=1)
    print(X.info())
    y = df['tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test



def train_model(df,train_length):
    df = df[:train_length]
    X_train, X_test, y_train, y_test = train_test(df)
    write_model(X_train, y_train)

def test_model(df,train_length):
    df = df[train_length:]
    X = df.drop(['tag', 'id', 'RS', 'High', 'Low', 'Price Change'], axis=1)
    # X_train, X_test, y_train, y_test = train_test(df)
    print(X)
    rf = read_model()
    y_pred = rf.predict(X)
    # print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    return y_pred.tolist()


if __name__ == '__main__':
    #df = pd.read_csv(r'/home/araz-abedini-bakhshmand/Documents/ai/test/XAUUSD_candlestick_1D.json', header=None)
    df = pd.read_json(r'/home/araz/Documents/ai/files/XAUUSD_candlestick_1D.json',convert_dates=True)
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
    total_tag, signal_list = plot_data_frame(date_list, price)
    df['tag'] = total_tag
    df = utility.normalize_col(col_name='Close', data_frame=df)
    train_length = int(len(df) * 0.8)
    train_model(df, train_length)
    test_tag = test_model(df, train_length)
    trained_tag = total_tag[-len(test_tag):]
    test_price = price[-len(test_tag):]
    date_list = date_list[-len(test_tag):]
    win_rate = win_rate_test.compute_win_rate(trained_tag, test_tag)
    print(win_rate)
    #print(risk_to_reward.risk_reward_compute(test_tag, test_price, date_list))
    date_list = date_list[-len(test_tag):]
    print(trained_tag)
    print(test_tag)
    print(date_list)
    func = limit.limit_fuction(test_tag, test_price)
    #print(risk_to_reward.risk_reward_compute(func, test_price, date_list))
    ploting(date_list, test_price, test_tag)
    utility.confusion_matrix(trained_tag, test_tag, ['B', 'S'])
