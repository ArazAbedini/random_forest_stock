from AlmaIndicator import ALMAIndicator
import utility as ut
import pandas as pd
import numpy as np
import xlsxwriter
import pickle



if __name__ == '__main__':
    df = pd.read_json('data.json', convert_dates=True)
    utility = ut.Utility()
    df = df.dropna()
    price = utility.make_list(df, 'Close')
    df = utility.alma_calculator(data_frame=df)
    df = df[1:]
    df = utility.calculate_rsi(df, price)
    tag_list = []
    diff_list = []
    df = df[-532:]
    print(len(df))
    df.dropna(inplace=True)
    date_list = utility.make_list(df, 'Open Time')
    df.drop('Open Time', inplace=True, axis=1)
    date_list = np.array(list(map(str, date_list)))
    price = utility.make_list(df, 'Close')
    print(df)
    with open('/home/araz/Documents/ai/stock/model/daily.obj', 'rb') as f:
        rfc = pickle.load(f)
    X = df.drop(['id', 'RS', 'Price Change'], axis=1)
    y_pred = rfc.predict(X)
    with open('tag.txt', 'r') as file:
        file_read = file.read()
        file_str = file_read[1:-1].replace(' ', '').split(',')
        label_list = [item[1:-1] for item in file_str]
        file.close()
    same_res = 0
    for i in range(len(label_list)):
        if label_list[i] == y_pred[i]:
            same_res += 1

    print(same_res / len(label_list))
