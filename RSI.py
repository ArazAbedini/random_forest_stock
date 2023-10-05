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
    x_axis = x_axis
    y_axis = y_axis
    list = signal.savgol_filter(y_axis,21, 3)
    smooth_peaks, _ = signal.find_peaks(x=list)
    main_peaks, _ = signal.find_peaks(x=y_axis)
    negated_list = [-x for x in y_axis]
    negated_smooth_list = [-x for x in list]
    main_valley, _ = signal.find_peaks(x=negated_list)
    min_smooth_peaks, _ = signal.find_peaks(x=negated_smooth_list)
    np_ax_axis = np.array(x_axis)
    np_ay_axis = np.array(y_axis)
    np_ay_smooth = np.array(list)
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
    col = 0
    for i in range(len(x_axis)):
        if i in min_points:
            flag = True
            red_list.append(i)
            tag_list.append('B')
            signal_list.append('B')
            buy = x_axis[i]
            buy_price = y_axis[i]
        elif i in max_points:
            green_list.append(i)
            if flag == True:
                signal_list.append('S')
                tag_list.append('S')
                if buy != None:
                    row += 1
            else:
                signal_list.append('N')
                tag_list.append('N')
            flag = False

        else:
            if flag == True:
                red_list.append(i)
                if i == 0:
                    signal_list.append('N')
                    tag_list.append('N')
                    flag = False
                else:
                    try:
                        risk_reward = (0.95 * buy_price) / (y_axis[i] - buy_price)
                        if risk_reward < 1.5:
                            signal_list.append('N')
                            tag_list.append('N')
                        else:
                            signal_list.append('S/B')
                            tag_list.append('S/B')
                            buy_price = y_axis[i]
                            row += 1
                            buy = x_axis[i]

                    except:
                        signal_list.append('N')
                        tag_list.append('N')
            else:
                green_list.append(i)
                signal_list.append('N')
                tag_list.append('N')
    ploting(x_axis,y_axis,tag_list)
    return tag_list, signal_list

def write_model(X_train, y_train):
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    with open('/home/araz-abedini-bakhshmand/Documents/ai/random_forest/model/daily.obj', 'wb') as f:
      pickle.dump(rfc, f)

def read_model():
    with open('/home/araz-abedini-bakhshmand/Documents/ai/random_forest/model/daily.obj', 'rb') as f:
        rfc = pickle.load(f)
    return rfc
def train_test(df):
    X = df.drop(['tag', 'Open', 'High', 'Low', 'id', 'RS'], axis=1)
    y = df['tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
    return X_train, X_test, y_train, y_test



def train_model(df,train_length):
    df = df[:train_length]
    X_train, X_test, y_train, y_test = train_test(df)
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    write_model(X_train,y_train)

def test_model(df,train_length):
    df = df[train_length:]
    X_train, X_test, y_train, y_test = train_test(df)
    rf = read_model()
    y_pred = rf.predict(X_test)
    print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    return y_pred.tolist()


if __name__ == '__main__':
    #df = pd.read_csv(r'/home/araz-abedini-bakhshmand/Documents/ai/test/XAUUSD_candlestick_1D.json', header=None)
    df = pd.read_json(r'/home/araz-abedini-bakhshmand/Documents/ai/test/XAUUSD_candlestick_1D.json',convert_dates=True)
    utility = ut.Utility()
    df = df.dropna()
    price = utility.make_list(df, 'Close')
    df = utility.alma_calculator(data_frame=df)
    diff_price = diff(price)
    df = df[1:]
    df['Price Change'] = diff_price
    period_length = 14
    df['Gain'] = np.where(df['Price Change'] > 0, df['Price Change'], 0)
    df['Loss'] = np.where(df['Price Change'] < 0, abs(df['Price Change']), 0)
    df['Avg Gain'] = df['Gain'].rolling(window=period_length).mean()
    df['Avg Loss'] = df['Loss'].rolling(window=period_length).mean()
    df['RS'] = df['Avg Gain'] / df['Avg Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    diff = diff(price)
    tag_list = []
    diff_list = []
    rsi = utility.make_list(df, 'alma')
    df = df.drop(['Avg Gain'], axis=1)
    df = df.drop(['Gain'], axis=1)
    df = df.drop(['Loss'], axis=1)
    df = df.drop(['Avg Loss'], axis=1)
    df = df.drop(df[df['RS'] == 'inf'].index)
    df.dropna(inplace=True)
    date_list = utility.make_list(df, 'Open Time')
    df.drop('Open Time', inplace=True, axis=1)
    date_list = list(map(str, date_list))
    price = utility.make_list(df, 'Close')
    x_array = np.array(price)
    normalized_arr = preprocessing.normalize([x_array])
    df['Close'] = normalized_arr[0]
    total_tag, signal_list = plot_data_frame(date_list, price)
    df['tag'] = total_tag
    df = df[13:]
    total_tag = total_tag[13:]
    train_length = int(len(df) * 0.8)
    train_model(df, train_length)
    test_tag = test_model(df, train_length)
    trained_tag = total_tag[-len(test_tag):]
    test_price = price[-len(test_tag):]
    date_list = date_list[-len(test_tag):]
    win_rate = win_rate_test.compute_win_rate(trained_tag, test_tag)
    print(win_rate)
    print(risk_to_reward.risk_reward_compute(test_tag, test_price, date_list))
    date_list = date_list[-len(test_tag):]
    print(trained_tag)
    print(test_tag)
    print(date_list)
    func = limit.limit_fuction(test_tag, test_price)
    print(risk_to_reward.risk_reward_compute(func, test_price, date_list))
    ploting(date_list, test_price, test_tag)
    cm = confusion_matrix(trained_tag, test_tag, labels=['B', 'S', 'S/B', 'N'])
    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=['B', 'S', 'S/B', 'N'],
                yticklabels=['B', 'S', 'S/B', 'N'])
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()
