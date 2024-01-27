from statsmodels.tsa.seasonal import STL
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
import pickle
import ast



def add_date_feature(df: pd.DataFrame) -> pd.DataFrame:
    time = np.array(df['open_time'])
    day_array = np.zeros(len(time), dtype=np.int8)
    month_array = np.zeros(len(time), dtype=np.int8)
    year_array = np.zeros(len(time), dtype=np.int8)
    hour_array = np.zeros(len(time), dtype=np.int8)
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
    X = df.drop(['id', 'open_time', 'time_frame', 'exchange', 'symbol'], axis='columns')[:-1]
    y = df['label'][1:]
    print(type(y))
    train_size = int(0.6 * len(X))
    cv_size = int(0.8 * len(X))
    train_model(X[:train_size], y[:train_size])
    with open('/home/araz/Documents/ai/stock/model/avax.obj', 'rb') as f:
        rfc = pickle.load(f)
    y_cv_predict = rfc.predict(X[train_size:cv_size])
    y_cv_actual = y[train_size:cv_size]
    accuracy = calculate_accuracy(y_cv_actual, y_cv_predict)
    print(f"Accuracy of train model : {accuracy * 100:.2f}%")
