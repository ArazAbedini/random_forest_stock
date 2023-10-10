from AlmaIndicator import ALMAIndicator
from sklearn import preprocessing
from numpy import diff
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix





class Utility:
    def __init__(self):
        pass

    @staticmethod
    def make_list(data_frame, col_name):
        column_array = data_frame[col_name].to_numpy()
        return column_array

    @staticmethod
    def delete_char(data_frame, col_name):
        for index, row in data_frame.iterrows():
            row[col_name] = row[col_name][:-1]
        return data_frame

    @staticmethod
    def remove_date(df, utility):
        date = utility.make_list(df, 'date')
        df.drop('date', inplace=True, axis=1)

    @staticmethod
    def title_label(data_frame):
        data_frame = data_frame.iloc[1:]
        col_names = ['date', 'close', 'open', 'high', 'low', 'volume', 'change']
        data_frame.columns = col_names
        return data_frame


    @staticmethod
    def rist_to_reward_ratio(x1, x2):
        return (0.95 * x1) / (x2 - x1)

    @staticmethod
    def tagg(list):
        tag_list = []
        for i in range(len(list)):
            if i == 0:
                tag_list.append(True)
                temp = list[i]
            else:
                try:
                    rr_ratio = ((0.95 * temp) / (list[i] - temp))
                    if rr_ratio >= 1.5:
                        tag_list.append(False)
                    else:
                        tag_list.append(True)
                        temp = list[i]
                except:
                    tag_list.append(False)
                    temp = list[i]

        return tag_list


    @staticmethod
    def change_type(data_frame):
        data_frame['close'] = data_frame['close'].astype('float')
        data_frame['open'] = data_frame['open'].astype('float')
        data_frame['high'] = data_frame['high'].astype('float')
        data_frame['low'] = data_frame['low'].astype('float')
        data_frame['volume'] = data_frame['volume'].astype('float')
        data_frame['change'] = data_frame['change'].astype('float')
        return data_frame


    @staticmethod
    def alma_calculator(data_frame):
        print(len(data_frame))
        alma_indicator = ALMAIndicator(close=data_frame['Close'])
        alma = alma_indicator.alma()
        print(len(alma))
        data_frame['alma'] = alma
        return data_frame


    @staticmethod
    def normalize_col(col_name, data_frame):
        x_array = data_frame[col_name].to_numpy()
        normalized_arr = preprocessing.normalize([x_array])
        data_frame[col_name] = normalized_arr[0]
        return data_frame


    @staticmethod
    def calculate_rsi(df, price):
        diff_price = diff(price)
        df['Price Change'] = diff_price
        period_length = 14
        df['Gain'] = np.where(df['Price Change'] > 0, df['Price Change'], 0)
        df['Loss'] = np.where(df['Price Change'] < 0, abs(df['Price Change']), 0)
        df['Avg Gain'] = df['Gain'].rolling(window=period_length).mean()
        df['Avg Loss'] = df['Loss'].rolling(window=period_length).mean()
        df['RS'] = df['Avg Gain'] / df['Avg Loss']
        df['RSI'] = 100 - (100 / (1 + df['RS']))
        #df = df.drop(['Price Change'], axis=1)
        df = df.drop(['Avg Gain'], axis=1)
        df = df.drop(['Gain'], axis=1)
        df = df.drop(['Loss'], axis=1)
        df = df.drop(['Avg Loss'], axis=1)
        df = df.drop(df[df['RS'] == 'inf'].index)
        return df

    @staticmethod
    def confusion_matrix(trained_tag, test_tag, labels):
        cm = confusion_matrix(trained_tag, test_tag, labels=labels)
        sns.heatmap(cm,
                    annot=True,
                    fmt='g',
                    xticklabels=labels,
                    yticklabels=labels)
        plt.ylabel('Actual', fontsize=13)
        plt.xlabel('Prediction', fontsize=13)
        plt.title('Confusion Matrix', fontsize=17)
        plt.show()