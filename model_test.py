from AlmaIndicator import ALMAIndicator
import utility as ut
import pandas as pd
import numpy as np
import xlsxwriter
import pickle



def read_file(utility):
    df = pd.read_csv(r'/home/araz-abedini-bakhshmand/Documents/ai/random forest/test.csv', header=None)
    df = utility.title_label(df)
    df = utility.delete_char(df, 'change')
    df = utility.delete_char(df, 'volume')
    df = utility.change_type(data_frame=df)
    df = df.loc[::-1].reset_index(drop=True)
    df = utility.alma_calculator(data_frame=df)
    return df


def add_feature(utility, df):
    from numpy import diff
    price = utility.make_list(df, 'close')
    diff_price = diff(price)
    df = df.iloc[1:]
    df = df.assign(price_change=diff_price)
    period_length = 14
    df = df.assign(Gain=np.where(df['price_change'] > 0, df['price_change'], 0))
    date_list = utility.make_list(df, 'date')
    df = df.assign(Loss=np.where(df['price_change'] < 0, abs(df['price_change']), 0))
    df = df.assign(Avg_Gain=df['Gain'].rolling(window=period_length).mean())
    df = df.assign(Avg_Loss=df['Loss'].rolling(window=period_length).mean())
    df = df.assign(RS=df['Avg_Gain'] / df['Avg_Loss'])
    df = df.assign(RSI=100 - (100 / (1 + df['RS'])))
    change = utility.make_list(df, 'change')
    price = list(map(float, price))
    df['close'] = df['close'].astype('float')
    tag_list = []
    change = utility.make_list(df, 'change')
    df['close'] = df['close'].astype('float')
    df = df.drop(['Avg_Gain'], axis=1)
    df = df.drop(['Gain'], axis=1)
    df = df.drop(['Loss'], axis=1)
    df = df.drop(['Avg_Loss'], axis=1)
    df = df[13:]
    return df



if __name__ == '__main__':
    utility = ut.Utility()
    df = read_file(utility)
    df = add_feature(utility, df)
    with open('/home/araz-abedini-bakhshmand/Documents/ai/random forest/model/random_forest.obj', 'rb') as f:
        rfc = pickle.load(f)
    list = utility.make_list(df,'price_change')
    date_list = utility.make_list(df, "date")
    df = df.drop("date", axis='columns')
    #df['Price Change'] = list
    df = df.drop(['price_change'],axis=1)
    df.insert(7, "Price Change", list, True)
    X = df.drop(['open', 'high', 'low'], axis=1)
    print(X.info())
    y_list = []
    for i in range(len(df['close'])):
        df2 = X.iloc[i:i+1]
        y_test = rfc.predict(df2)
        y_list.append(y_test[0])
    workbook = xlsxwriter.Workbook('test.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    for i in range(len(list)):
        worksheet.write(row, 0, date_list[i])
        worksheet.write(row, 1, y_list[i])
        row += 1
    workbook.close()
