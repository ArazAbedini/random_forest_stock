from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from AlmaIndicator import ALMAIndicator
from difference import mean_calculate, cut_signal, find_small_difference, find_big_difference
from sklearn import preprocessing
from stoploss import mean_diff_stoploss, end_point
from regression_torch import regression_train, regression_test
import matplotlib.pyplot as plt
from scipy import signal
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
    print('y_axis : ', y_axis)
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
    feature_importance = rfc.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)
    print('opopopopopopopoppopopopopopopopopopopopopopopop')
    print(X_train, y_train)
    with open('/home/araz/Documents/ai/stock/model/daily.obj', 'wb') as f:
      pickle.dump(rfc, f)

def read_model():
    with open('/home/araz/Documents/ai/stock/model/daily.obj', 'rb') as f:
        rfc = pickle.load(f)
    # plt.figure(figsize=(20, 10))
    # plot_tree(rfc.estimators_[0], filled=True)
    # plt.savefig("decision_tree.png", dpi=720)
    # plt.show()
    return rfc
def train_test(df):
    X = df.drop(['tag', 'id', 'RS', 'Price Change', 'High', 'Close', 'Low', 'Open', '26-day EMA'], axis=1)
    print(X)
    correlation_matrix = X.corr()
    print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
    print(correlation_matrix)
    y = df['tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00001, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test



def train_model(df,train_length):
    print(train_length)
    df2 = df[:train_length]
    print('//////////////////////////////////////////////////////////////////////////////////////')
    print(df2)
    X_train, X_test, y_train, y_test = train_test(df2)
    write_model(X_train, y_train)

def test_model(df,train_length):
    df = df[train_length:]
    X = df.drop(['tag', 'id', 'RS', 'Price Change', 'High', 'Close', 'Low', 'Open', '26-day EMA'], axis=1)
    # X_train, X_test, y_train, y_test = train_test(df)
    print('???????????????????????????????????????????????????????????????????????????????????????')
    print(len(df))
    print(X)
    rf = read_model()
    y_pred = rf.predict(X)
    return y_pred.tolist()


def excel_write(date_list: list, price: list, take_profit: list, stop_loss: list, colors: list):
    current_state = None
    state_list = []
    # colors_diff = np.diff(np.array(colors))
    for i in range(len(price)):
        if (not current_state == 'buy') and colors[i] == 'green':
            current_state = 'buy'
            state_list.append('buy')
        elif current_state == 'buy' and colors[i] == 'red':
            state_list.append('sell')
            current_state = 'sell'
        else:
            state_list.append('nothing')
    data = {
        'date': date_list,
        'price': price,
        'stoploss': stop_loss,
        'takeprofit': take_profit,
        'signal': state_list
    }
    index = None
    sell_date = []
    sell_price = []
    good = 0
    bad = 0
    for i in range(len(state_list)):
        if state_list[i] == 'buy':
            index = i
            sell_date.append('-')
            sell_price.append('-')
        elif state_list[i] == 'sell':
            sell_date.append('-')
            sell_price.append('-')
            sell_date[index] = date_list[i]
            sell_price[index] = price[i]
        else:
            sell_date.append('-')
            sell_price.append('-')
    data = {
        'date': date_list,
        'price': price,
        'stoploss': stop_loss,
        'takeprofit': take_profit,
        'signal': state_list,
    }
    good = 0
    bad = 0
    for i in range(len(sell_price)):
        if sell_price[i] != '-':
            current_price = float(sell_price[i]) - float(price[i])
            if current_price > 0:
                good += 1
            else:
                bad += 1
    print('win rate is : ', good / (good + bad))
    df = pd.DataFrame(data)
    signals = np.array(df['signal'])
    stolosses = np.array(df['stoploss'])
    takeprofits = np.array(df['takeprofit'])
    prices = np.array(df['price'])
    dates = np.array(df['date'])
    end_time = []
    end_condition = []
    end_price = []
    for i in range(len(signals)):
        if signals[i] == 'buy':
            for j in range(i + 1,len(prices)):
                if stolosses[i] >= prices[j]:
                    end_time.append(str(dates[j]))
                    end_condition.append(str(0))
                    end_price.append(str(prices[j]))
                    break
                elif takeprofits[i] <= prices[j]:
                    end_time.append(str(dates[j]))
                    end_condition.append(str(1)) #succesfull
                    end_price.append(str(prices[j]))
                    break
        elif signals[i] == 'sell':
            for j in range(i + 1, len(prices)):
                if stolosses[i] <= prices[j]:
                    end_time.append(str(dates[j]))
                    end_condition.append(str(0))
                    end_price.append(str(prices[j]))
                    break
                elif takeprofits[i] >= prices[j]:
                    end_time.append(str(dates[j]))
                    end_condition.append(str(1))
                    end_price.append(str(prices[j]))
                    break
        else:
            end_time.append(' ')
            end_condition.append(' ')
            end_price.append(' ')
    # end_time.append(' ')
    # end_condition.append(' ')
    # end_price.append(' ')
    df['endTime'] = end_time
    df['endCondition'] = end_condition
    df['endPrice'] = end_price
    excel_file_path = 'daily.xlsx'
    good = 0
    bad = 0
    for i in range(len(end_condition)):
        if end_condition[i] == '1':
           good += 1
        elif end_condition[i] == '0':
            bad += 1
    print('------------------------')
    print(good / (good + bad))
    print(good + bad)
    print('------------------------')
    df.to_excel(excel_file_path, index=False)





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
    df['12-day EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26-day EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD Line'] = df['12-day EMA'] - df['26-day EMA']
    df['Signal Line'] = df['MACD Line'].ewm(span=9, adjust=False).mean()
    df['MACD Histogram'] = df['MACD Line'] - df['Signal Line']
    total_tag, signal_list = plot_data_frame(date_list, price)
    df['tag'] = total_tag
    print(df)
    y = np.array(df['Close'][1:])
    X = df.drop(['id', 'tag'], axis='columns')[:-1]
    print(X)
    print('./././././././././././././././././././././././.')
    X_train = X
    y_train = y
    regression_train()
    print(total_tag)
    test_tag1 = total_tag[train_length:]
    train_model(df, train_length)
    test_tag = test_model(df, train_length)
    trained_tag = total_tag[-len(test_tag):]
    test_price = price[-len(test_tag):]
    test_date_list = date_list[-len(test_tag):]
    price_arr = np.array(price)
    stoploss_arr = end_point(price, total_tag)
    stoploss_diff = stoploss_arr - price_arr
    take_profit_arr = 2 * -1 * stoploss_diff + price_arr
    print('stoploss array is : ', stoploss_arr)
    print('take profit arr is : ', take_profit_arr)
    ploting(test_date_list, test_price, test_tag)
    utility.confusion_matrix(trained_tag, test_tag, ['B', 'S'])
    colors = []
    for index in test_tag:
        if index == 'B':
            colors.append('green')
        elif index == 'S':
            colors.append('red')
        elif index == 'S/B':
            colors.append('orange')
        else:
            colors.append('black')
    plt.plot(test_date_list, test_price)
    plt.scatter(test_date_list, test_price, c=colors)
    plt.xticks(rotation=90)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('total labeling')
    stoploss_arr = end_point(test_price, test_tag)
    stoploss_diff = stoploss_arr - test_price
    take_profit_arr = 2 * -1 * stoploss_diff + test_price
    print('stoploss array is : ', stoploss_arr)
    print('take profit arr is : ', take_profit_arr)
    plt.plot(test_date_list, stoploss_arr)
    plt.plot(test_date_list, take_profit_arr)
    plt.show()
    excel_write(test_date_list, test_price, take_profit_arr[-len(test_tag):], stoploss_arr[-len(test_tag):], colors)
    