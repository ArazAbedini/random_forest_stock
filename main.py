from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import diff
import pandas as pd
import utility as ut
import tag as t

# import matplotlib.pyplot as plt


# True mean buy

if __name__ == '__main__':
    df = pd.read_csv(r'/home/araz-abedini-bakhshmand/Documents/ai/random forest/AAPL Historical Data.csv', header=None)
    df = df.iloc[1:]
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']
    df.columns = col_names
    utility = ut.Utility()
    date = utility.make_list(df, 'date')
    df = utility.delete_char(df, 'change')
    df = utility.delete_char(df, 'volume')
    df.drop('date', inplace=True, axis=1)
    change = utility.make_list(df, 'change')
    change = list(map(float, change))
    price = utility.make_list(df, 'close')
    price = list(map(float, price))
    date.reverse()
    change.reverse()
    price.reverse()
    # change = change[:50]
    # price = price[:50]
    diff = diff(price) / 1
    tag_list = []
    for i in range(len(change)):
        result = True if not change[i - 3:i] else False
        # print(diff[i - 3:i])
        historical_change = diff[i - 3:i]
        tag = t.Tag(historical_change)
        result = tag.tag_point() if not result else result
        tag_list.append(result)
    # date = date[:50]
    df['tag'] = tag_list
    df['open'] = df['open'].astype('float')
    df['high'] = df['high'].astype('float')
    df['low'] = df['low'].astype('float')
    df['volume'] = df['volume'].astype('float')
    df['change'] = df['change'].astype('float')
    df['close'] = df['close'].astype('float')
    print(df)
    print(df.dtypes)
    X = df.drop(['tag'], axis=1)
    y = df['tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    # plt.subplot(1, 2, 1)
    # plt.plot(date, change, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue',
    #          markersize=12)
    # plt.xticks(rotation=90)
    # plt.subplot(1, 2, 2)
    # plt.plot(date, price, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue',
    #          markersize=12)
    # plt.xticks(rotation=90)
    # plt.show()
    plt.plot(date, price)
    plt.show()
    # print(df.head())
