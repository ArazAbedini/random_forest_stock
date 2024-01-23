from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
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

def label_process(close: np.ndarray) -> np.ndarray:
    fig, ax = plt.subplots()
    smooth_arr = signal.savgol_filter(close, window_length=9, polyorder=3)
    smooth_peak, _ = signal.find_peaks(x=smooth_arr)
    smooth_peak = find_actual_peak(smooth_peak, close)
    main_peak, _ = signal.find_peaks(x=close)
    main_valley, _ = signal.find_peaks(x=-1 * close)
    print(f'the length of main_peak is {len(main_peak)} and the length of main_valley is {len(main_valley)}')
    print(f'length of smooth arr is {len(smooth_peak)}')
    smooth_valley, _ = signal.find_peaks(x=-1 * smooth_peak)
    moving_average = df['close'].rolling(7).mean().shift(-2)
    ax.plot(close[smooth_peak], color='red')
    ax.plot(close, color='blue')
    # ax.plot(moving_average, color='red')
    # ax.plot(close, color='black')
    # ax.plot(smooth_arr, color='blue')


    plt.show()

def train_model(df: pd.DataFrame):
    X = df.drop(['id', 'open_time', 'time_frame', 'exchange', 'symbol'], axis='columns')[:-1]
    y = df['close'].shift(1)[1:]


if __name__ == '__main__':
    FILE_PATH = '/home/araz/Documents/ai/files/AVAX_1h.txt'
    with open(FILE_PATH, 'r') as file:
        content = file.read()
    content_list = ast.literal_eval(content)
    df = pd.DataFrame(content_list)
    df = add_date_feature(df)
    X = df.drop(['id', 'open_time', 'time_frame', 'exchange', 'symbol'], axis='columns')[:-1]
    print(X)
    columns_list = ['open', 'close', 'high', 'low', 'volume']
    df[columns_list] = df[columns_list].astype(np.float32)
    close_array = np.array(df['close'])
    label_process(close_array)

