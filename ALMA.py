from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from AlmaIndicator import ALMAIndicator
from numpy import diff
import pandas as pd
import utility as ut
import tag as t



if __name__ == '__main__':

    df = pd.read_csv(r'/home/araz-abedini-bakhshmand/Documents/ai/random forest/AAPL Historical Data.csv', header=None)
    df = df.iloc[1:]
    col_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']
    df.columns = col_names
    utility = ut.Utility()
    df = utility.delete_char(df, 'change')
    df = utility.delete_char(df, 'volume')
    df['close'] = df['close'].astype('float')
    alma_indicator = ALMAIndicator(close=df['close'])
    df['alma'] = alma_indicator.alma()
    change = utility.make_list(df, 'change')
    change = list(map(float, change))
    price = utility.make_list(df, 'close')
    price = list(map(float, price))
    diff = diff(price) / 1
    tag_list = []

    for i in range(len(change)):
        result = True if not change[i - 3:i] else False
        historical_change = diff[i - 3:i]
        tag = t.Tag(historical_change)
        result = tag.tag_point() if not result else result
        tag_list.append(result)
    utility.remove_date(df,utility)
    df['tag'] = tag_list
    df = df.iloc[8:]
    X = df.drop(['tag'], axis=1)
    y = df['tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))



