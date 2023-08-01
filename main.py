import pandas as pd
# import matplotlib.pyplot as plt
from numpy import diff


def delete_percent(data_frame):
    for index, row in data_frame.iterrows():
        row['change'] = row['change'][:-1]
    return data_frame


def make_list(data_frame, col_name):
    output = []
    for index, row in data_frame.iterrows():
        output.append(row[col_name])
    return output


# True mean buy


def positive_number(historical_changes):
    if all(i < j for i, j in zip(historical_changes, historical_changes[1:])):
        res = True
    elif all(i > j for i, j in zip(historical_changes, historical_changes[1:])):
        res = False if historical_changes[2] < 0.5 else True
    else:
        res = True
    return res


def negative_number(historical_changes):
    if all(i > j for i, j in zip(historical_changes, historical_changes[1:])):
        res = False
    elif all(i < j for i, j in zip(historical_changes, historical_changes[1:])):
        res = True if abs(historical_changes[2]) < 0.5 else False
    else:
        res = False
    return res


def complicated_number(historical_changes):
    if historical_changes[0] > 0:
        if historical_changes[1] < 0:
            res = True if historical_changes[2] > 0 else False
        else:
            res = False if abs(historical_changes[1]) < 0.5 else True
    else:
        if historical_changes[1] > 0:
            res = True if historical_changes[2] > 0 else False
        else:
            res = True if abs(historical_changes[1]) < 0.5 else False
    return res


def tag_point(historical_changes):
    res = all(i > 0 and j > 0 for i, j in
              zip(historical_changes, historical_changes[1:]))  # res = True if it means that we have to buy
    if res:
        res = positive_number(historical_changes)
    elif all(i < 0 and j < 0 for i, j in zip(historical_changes, historical_changes[1:])):
        res = negative_number(historical_changes)
    else:
        res = complicated_number(historical_changes)
    return res


if __name__ == '__main__':

    df = pd.read_csv(r'/home/araz-abedini-bakhshmand/Documents/ai/random forest/AAPL Historical Data.csv', header=None)
    df = df.iloc[1:]
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']
    df.columns = col_names
    date = make_list(df, 'date')
    df = delete_percent(df)
    df.drop('date', inplace=True, axis=1)
    change = make_list(df, 'change')
    change = list(map(float, change))
    price = make_list(df, 'close')
    price = list(map(float, price))
    date.reverse()
    change.reverse()
    price.reverse()
    #change = change[:50]
    #price = price[:50]
    diff = diff(price) / 1
    tag_list = []
    for i in range(len(change)):
        result = True if not change[i - 3:i] else False
        #print(diff[i - 3:i])
        historical_change = diff[i - 3:i]
        result = tag_point(historical_change) if not result else result
        tag_list.append(result)
    #date = date[:50]
    df['tag'] = tag_list
    # plt.subplot(1, 2, 1)
    # plt.plot(date, change, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue',
    #          markersize=12)
    # plt.xticks(rotation=90)
    # plt.subplot(1, 2, 2)
    # plt.plot(date, price, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue',
    #          markersize=12)
    # plt.xticks(rotation=90)
    # plt.show()

    # print(df.head())
