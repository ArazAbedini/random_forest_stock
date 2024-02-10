from statsmodels.tsa.seasonal import STL
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from stoploss import end_point
from RSI import excel_write
from scipy import signal
import pandas as pd
import numpy as np
import pickle
import ast



def add_date_feature(df: pd.DataFrame) -> pd.DataFrame:
    time = np.array(df['open_time'])
    day_array = np.zeros(len(time), dtype=np.int16)
    month_array = np.zeros(len(time), dtype=np.int16)
    year_array = np.zeros(len(time), dtype=np.int16)
    hour_array = np.zeros(len(time), dtype=np.int16)
    for i in range(len(time)):
        time_list = time[i].split(' ')
        date_list = time_list[0].split('-')
        hour_array[i] = time_list[1].split(':')[0]
        year_array[i] = date_list[0]
        month_array[i] = date_list[1]
        day_array[i] = date_list[2]

    feature_df = pd.DataFrame({
        'year': year_array,
        'month': month_array,
        'day': day_array,
        'hour': hour_array
    })

    df_result = pd.concat([df, feature_df], axis='columns')
    return df_result


def find_actual_peak(smooth_peak: np.ndarray, close: np.ndarray) -> np.ndarray:
    TOLERANCE = 5
    output_arr = np.zeros(len(smooth_peak), dtype=np.int32)
    for i in range(len(smooth_peak)):
        start = max(smooth_peak[i] - 5, 0)
        end = min(smooth_peak[i] + 5, len(close))
        max_index_close = np.argmax(close[start: end])
        output_arr[i] = max_index_close + start
    output = np.unique(output_arr)
    return output

def find_actual_valleys(smooth_peak: np.ndarray, close: np.ndarray) -> np.ndarray:
    TOLERANCE = 10
    output_arr = np.zeros(len(smooth_peak), dtype=np.int32)
    for i in range(len(smooth_peak)):
        start = max(smooth_peak[i] - 5, 0)
        end = min(smooth_peak[i] + 5, len(close))
        min_index_close = np.argmin(close[start: end])
        output_arr[i] = min_index_close + start
    output = np.unique(output_arr)
    return output

def label_process(close: np.ndarray, open_time: np.ndarray) -> np.ndarray:
    fig, ax = plt.subplots()
    smooth_arr = signal.savgol_filter(close, window_length=14, polyorder=2)
    smooth_peak, _ = signal.find_peaks(x=smooth_arr)
    smooth_valleys, _ = signal.find_peaks(x=-1 * smooth_arr)
    smooth_peak = find_actual_peak(smooth_peak, close) # in this function we have index of peaks
    smooth_valleys = find_actual_valleys(smooth_valleys, close) # in this function we have index of valleys
    y_peaks = close[smooth_peak]
    y_valleys = close[smooth_valleys]
    label = np.zeros(len(close), dtype=np.int32)
    state = 0
    for i in range(len(close)):
        if i in smooth_peak:
            state = 1
            label[i] = 1
        elif i in smooth_valleys:
            state = 0
        elif state == 1:
            label[i] = 1

    # ax.plot(open_time, close, color='black')
    # ax.plot(open_time[smooth_peak], y_peaks, 'ro')
    # ax.plot(open_time[smooth_valleys], y_valleys, 'go')
    # plt.xticks(rotation=90)
    print(f'length of the smooth peak is {len(smooth_peak)}')
    print(f'length of close is {len(close)}')
    # plt.show()
    return label

def calculate_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    correct_predictions = np.sum(actual == predicted)
    total_predictions = len(actual)
    accuracy = correct_predictions / total_predictions
    return accuracy
def train_model(X: pd.DataFrame, y: pd.Series):
    xgb_model = XGBClassifier(
        objective='multi:softmax',  # for multi-class classification
        num_class=len(np.unique(y)),
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        n_estimators=100,
        random_state=42
    )
    xgb_model.fit(X, y)
    y_actual = np.array(y)
    y_predict = xgb_model.predict(X)
    accuracy = calculate_accuracy(y_actual, y_predict)
    print(f"Accuracy of train model : {accuracy * 100:.2f}%")
    with open('/home/araz/Documents/ai/stock/model/avax.obj', 'wb') as f:
        pickle.dump(xgb_model, f)

if __name__ == '__main__':
    FILE_PATH = '/home/araz/Documents/ai/files/AVAX_1h.txt'
    with open(FILE_PATH, 'r') as file:
        content = file.read()
    content_list = ast.literal_eval(content)
    df = pd.DataFrame(content_list)
    df = add_date_feature(df)
    open_time = df['open_time']
    close_array = np.array(df['close'])
    label = label_process(close_array, open_time)
    df['label'] = label
    columns_list = ['open', 'close', 'high', 'low', 'volume']
    df[columns_list] = df[columns_list].astype(np.float32)
    X = df.drop(['id', 'open_time', 'time_frame', 'exchange', 'symbol', 'label'], axis='columns')[:-1]
    y = df['label'][1:]
    train_size = int(0.6 * len(X))
    cv_size = int(0.8 * len(X))
    train_model(X[:train_size], y[:train_size])
    with open('/home/araz/Documents/ai/stock/model/avax.obj', 'rb') as f:
        rfc = pickle.load(f)
    y_cv_predict = rfc.predict(X[train_size:cv_size])
    y_cv_actual = y[train_size:cv_size]
    accuracy = calculate_accuracy(y_cv_actual, y_cv_predict)
    print(f"Accuracy of train model : {accuracy * 100:.2f}%")
    x_test = X[cv_size:]
    y_test_predict = rfc.predict(x_test)
    y_test_actual = y[cv_size:]
    accuracy = calculate_accuracy(y_test_actual, y_test_predict)
    print(f"Accuracy of train model : {accuracy * 100:.2f}%")
    date_excel = np.array(open_time[cv_size:-1])
    close_excel = np.array(x_test['close'])
    stoploss_arr = end_point(list(close_excel), list(y_test_predict))
    stoploss_diff = stoploss_arr - close_excel
    take_profit_arr = 2 * -1 * stoploss_diff + close_excel
    colors = []
    for item in y_test_predict:
        if item == 0:
            colors.append('green')
        elif item == 1:
            colors.append('red')
        else:
            colors.append('black')


    excel_write(list(date_excel), list(close_excel), list(take_profit_arr), list(stoploss_arr), list(colors), name='avax_daily.xlsx')
    # day_array = np.full(len(close_excel), None, dtype=object)
    # state_array = np.full(len(close_excel), 'undifined', dtype=object)
    # end_price = np.zeros(len(close_excel), dtype=np.int16)
    # for i in range(len(close_excel)):
    #     state = y_test_predict[i]
    #     for j in range(i + 1, len(close_excel)):
    #         if close_excel[j] <= stoploss_arr[i] and state == 0:
    #             day_array[i] = date_excel[j]
    #             end_price[i] = close_excel[j]
    #             state_array[i] = 'fail'
    #             break
    #         elif close_excel[j] >= take_profit_arr[i] and state == 0:
    #             day_array[i] = date_excel[j]
    #             state_array[i] = 'successful'
    #             end_price[i] = close_excel[j]
    #             break
    #         elif close_excel[j] <= take_profit_arr[i] and state == 1:
    #             day_array[i] = date_excel[j]
    #             state_array[i] = 'successful'
    #             end_price[i] = close_excel[j]
    #             break
    #         elif close_excel[j] >= stoploss_arr[i] and state == 1:
    #             day_array[i] = date_excel[j]
    #             end_price[i] = close_excel[j]
    #             state_array[i] = 'fail'
    #             break
    #
    # data = {
    #     'time': date_excel,
    #     'close': close_excel,
    #     'take profit': take_profit_arr,
    #     'stoploss': stoploss_arr,
    #     'signal': y_test_predict,
    #     'end time': day_array,
    #     'end price': end_price,
    #     'state': state_array
    # }
    # df = pd.DataFrame(data)
    # df.to_excel('avax.xlsx', index=False)
    # success = 0
    # fail = 0
    # for state in state_array:
    #     if state == 'successful':
    #         success += 1
    #     elif state == 'fail':
    #         fail += 1
    # print(success / len(state_array))
