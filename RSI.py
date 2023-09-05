from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from AlmaIndicator import ALMAIndicator
import matplotlib.pyplot as plt
from scipy import signal
from numpy import diff
import win_rate as wr
import utility as ut
import pandas as pd
import numpy as np
import xlsxwriter
import tag as t
import pickle




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
    # plt.scatter(x=x_axis, y=y_axis, c=colors_point)
    list = signal.savgol_filter(y_axis,21, 3)
    plt.plot(x_axis, list)
    plt.plot(x_axis, y_axis)
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
    plt.plot(np_ax_axis[max_points], np_ay_axis[max_points], 'ko', label='Peaks')
    plt.plot(np_ax_axis[smooth_peaks], np_ay_smooth[smooth_peaks], 'bo', label='Smooth')
    plt.plot(np_ax_axis[min_smooth_peaks], np_ay_smooth[min_smooth_peaks], 'yo', label='Smooth')
    plt.plot(np_ax_axis[min_points], np_ay_axis[min_points], 'mo', label='Valleys')
    tag_list = []
    red_list = []
    green_list = []
    flag = True
    signal_list = []
    buy_price = None
    for i in range(len(x_axis)):
        if i in min_points:
            flag = True
            red_list.append(i)
            tag_list.append('B')
            signal_list.append('B') #completed!
            buy_price = y_axis[i]
        elif i in max_points:
            green_list.append(i)
            if flag == True:
                signal_list.append('S')
                tag_list.append('S')
            else:                       # completed!
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
                    except:
                        signal_list.append('N')
                        tag_list.append('N')
            else:
                green_list.append(i)
                signal_list.append('N')
                tag_list.append('N')
    plt.plot(np_ax_axis[red_list], np_ay_axis[red_list], 'ro', label='valleys')
    plt.plot(np_ax_axis[green_list], np_ay_axis[green_list], 'go', label='peaks')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
    return tag_list, signal_list




if __name__ == '__main__':
    #df = pd.read_csv(r'/home/araz-abedini-bakhshmand/Documents/ai/test/XAUUSD_candlestick_1D.json', header=None)
    df = pd.read_json(r'/home/araz-abedini-bakhshmand/Documents/ai/test/XAUUSD_candlestick_1D.json',convert_dates=True)
    utility = ut.Utility()
    # df = utility.title_label(df)
    # df = utility.delete_char(df, 'change')
    # df = utility.delete_char(df, 'volume')
    # df = utility.change_type(data_frame=df)

    price = utility.make_list(df, 'Close')
    #date_list = utility.make_list(df, 'date')
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
    date_list = utility.make_list(df, 'Open Time')
    price = price[1:]
    tag = t.Tag(price)
    df.drop('Open Time', inplace=True, axis=1)
    date_list = list(map(str, date_list))
    rsi = utility.make_list(df, 'alma')
    tag_list = tag.tag_peak()
    tag_list = tag_list[1:]
    df = df.drop(['Avg Gain'], axis=1)
    df = df.drop(['Gain'], axis=1)
    df = df.drop(['Loss'], axis=1)
    df = df.drop(['Avg Loss'], axis=1)
    tagg, signal_list = plot_data_frame(date_list, price)
    tagg = tagg[15:]
    print(df)
    df = df[14:]
    print(df)
    df['tag'] = tagg
    process_win_rate = wr.WinRate(decision_list=signal_list, price_list=price)
    process_win_rate.calculate_win_rate()
    wind_rate = process_win_rate.win_rate()
    X = df.drop(['tag', 'open', 'high', 'low'], axis=1)
    y = df['tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    # with open('/home/araz-abedini-bakhshmand/Documents/ai/random forest/model/random_forest.obj', 'wb') as f:
    #   pickle.dump(rfc, f)
    with open('/home/araz-abedini-bakhshmand/Documents/ai/random forest/model/random_forest.obj', 'rb') as f:
        rfc = pickle.load(f)
    y_pred = rfc.predict(X_test)
    #print(date_list)
    date_list = date_list[15:]
    workbook = xlsxwriter.Workbook('test.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    for i in range(len(date_list)):
        worksheet.write(row, 0, date_list[i])
        worksheet.write(row, 1, tagg[i])
        row += 1
    workbook.close()
    print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

