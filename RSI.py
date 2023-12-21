import pandas
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
# import xgboost as xgb
from numpy import diff
from scipy import signal
import utility as ut
import win_rate_test
import pandas as pd
import numpy as np
import xlsxwriter
import tag as t
import pickle
import limit



def calculate_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    correct_predictions = np.sum(actual == predicted)
    total_predictions = len(actual)
    accuracy = correct_predictions / total_predictions
    return accuracy

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
    return tag_list, signal_list


def train_model(df: pd.DataFrame, train_length: int):
    df_train = df[:train_length]
    X = df_train.drop(['tag', 'id', 'Open Time', 'Close', 'High', 'Low', 'Open', '26-day EMA'], axis='columns')
    correlation_matrix = X.corr()
    print(correlation_matrix)
    y = df_train['tag']
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X, y)
    feature_importance = rfc.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)
    y_actual = np.array(y)
    y_predict = rfc.predict(X)
    accuracy = calculate_accuracy(y_actual, y_predict)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    with open('/home/araz/Documents/ai/stock/model/daily.obj', 'wb') as f:
        pickle.dump(rfc, f)

def cv_model(df: pd.DataFrame, train_length: int) -> np.ndarray:
    df_cv_test = df[train_length:]
    df_cv = df_cv_test[:len(df_cv_test) // 2]
    X = df_cv.drop(['tag', 'id', 'Open Time', 'Close', 'High', 'Low', 'Open', '26-day EMA'], axis='columns')
    y_actual = np.array(df_cv['tag'])
    correlation_matrix = X.corr()
    print(correlation_matrix)
    y = np.array(df_cv['tag'])
    with open('/home/araz/Documents/ai/stock/model/daily.obj', 'rb') as f:
        rfc = pickle.load(f)
    y_pred = rfc.predict(X)
    accuracy = calculate_accuracy(y_actual, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    feature_importance = rfc.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)
    return y_pred
def test_model(df: pd.DataFrame, train_length: int) -> np.ndarray:
    df_cv_test = df[train_length:]
    df_test = df_cv_test[len(df_cv_test) // 2:]
    X = df_test.drop(['tag', 'id', 'Open Time', 'Close', 'High', 'Low', 'Open', '26-day EMA'], axis='columns')
    y_actual = df_test['tag']
    with open('/home/araz/Documents/ai/stock/model/daily.obj', 'rb') as f:
        rfc = pickle.load(f)
    y_pred = rfc.predict(X)
    accuracy = calculate_accuracy(y_actual, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return y_pred

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
    print(len(data['date']))
    print(len(data['price']))
    print(len(data['stoploss']))
    print(len(data['takeprofit']))
    print(len(data['signal']))
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


def feature_add(df: pd.DataFrame) -> pd.DataFrame:
    alma_indicator = ALMAIndicator(close=df['Close'])
    alma = alma_indicator.alma()
    df['alma'] = alma
    diff_price = diff(price)
    diff_price = np.insert(diff_price, 0, None)
    print(diff_price.dtype)
    df['Price Change'] = diff_price
    period_length = 14
    df['Gain'] = np.where(df['Price Change'] > 0, df['Price Change'], 0)
    df['Loss'] = np.where(df['Price Change'] < 0, abs(df['Price Change']), 0)
    df['Avg Gain'] = df['Gain'].rolling(window=period_length).mean()
    df['Avg Loss'] = df['Loss'].rolling(window=period_length).mean()
    df['RS'] = df['Avg Gain'] / df['Avg Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    df = df.drop(['Avg Gain'], axis=1)
    df = df.drop(['Gain'], axis=1)
    df = df.drop(['Loss'], axis=1)
    df = df.drop(['Avg Loss'], axis=1)
    df = df.drop(df[df['RS'] == 'inf'].index)
    df['12-day EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26-day EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD Line'] = df['12-day EMA'] - df['26-day EMA']
    df['Signal Line'] = df['MACD Line'].ewm(span=9, adjust=False).mean()
    df['MACD Histogram'] = df['MACD Line'] - df['Signal Line']
    df.dropna(axis=1, inplace=True)
    return df


if __name__ == '__main__':
    df = pd.read_json(r'/home/araz/Documents/ai/files/XAUUSD_candlestick_1D.json',convert_dates=True)
    train_length = int(len(df) * 0.6)
    df_cross_test = df[train_length - 15:]
    cross_test_length = len(df_cross_test)
    df_cross_val = df_cross_test[:cross_test_length]
    df_test = df_cross_test[cross_test_length:]
    utility = ut.Utility()
    price = df['Close']
    df = feature_add(df)
    date_list = np.array(df['Open Time'])
    date_list = np.array(list(map(str, date_list)))
    total_tag, signal_list = plot_data_frame(date_list, price)
    df['tag'] = total_tag
    train_model(df, train_length)
    cv_tag = cv_model(df, train_length)
    test_tag = test_model(df, train_length)
    price_arr = np.array(price)
    stoploss_arr = end_point(price, total_tag)
    stoploss_diff = stoploss_arr - price_arr
    print('length of stoploss arr is : ', len(stoploss_arr))
    take_profit_arr = 2 * -1 * stoploss_diff + price_arr
    prediction_tag = np.concatenate((cv_tag, test_tag))
    print(len(prediction_tag))
    print(len(cv_tag))
    utility.confusion_matrix(df[train_length:]['tag'], prediction_tag, ['B', 'S'])
    colors = []
    for index in prediction_tag:
        if index == 'B':
            colors.append('green')
        elif index == 'S':
            colors.append('red')
        elif index == 'S/B':
            colors.append('orange')
        else:
            colors.append('black')
    prediction_price = df[train_length:]['Close']
    prediction_date = df[train_length:]['Open Time']
    stoploss_arr = end_point(list(prediction_price), list(prediction_tag))
    stoploss_diff = stoploss_arr - prediction_price
    take_profit_arr = 2 * -1 * stoploss_diff + prediction_price
    plt.plot(prediction_date, stoploss_arr)
    plt.plot(prediction_date, take_profit_arr)
    plt.show()
    print('length of take profit arr is : ', len(take_profit_arr))
    print('length of predction date is : ', len(prediction_date))
    excel_write(list(prediction_date), list(prediction_price), list(take_profit_arr), list(stoploss_arr), colors)
