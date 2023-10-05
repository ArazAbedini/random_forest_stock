import pandas as pd

class WinRate:
    def __init__(self, decision_list, price_list):
        self.buy_price = None
        self.success_trade = 0
        self.total_trade = 0
        self.decision_list = decision_list
        self.price_list = price_list

    def calculate_win_rate(self):
        for index in range(len(self.decision_list)):
            if self.decision_list[index] == 'B':
                self.buy_price = self.price_list[index]
            elif self.decision_list[index] == 'S' and self.buy_price is not None:
                self.total_trade += 1
                self.sell_price = self.price_list[index]
                if self.sell_price > self.buy_price:
                    self.success_trade += 1
                self.buy_price = None
            elif self.decision_list[index] == 'S/B' and self.buy_price is not None:
                self.total_trade += 1
                self.sell_price = self.price_list[index]
                if self.sell_price > self.buy_price:
                    self.success_trade += 1
                self.buy_price = self.price_list[index]
            else:
                pass

    def win_rate(self):
        return self.success_trade / self.total_trade



if __name__ == '__main__':
    dataframe = pd.read_excel('Example2.xlsx')
    dataframe = dataframe[2159:]
    dataframe.columns = ['date', 'test', 'total']
    print(dataframe)
    test = []
    for index, row in dataframe.iterrows():
        test.append(row['test'])
    total = []
    for index, row in dataframe.iterrows():
        total.append(row['total'])
    error = 0
    count = 0
    for i in range(len(total)):
        count += 1
        if total[i] != test[i]:
            error += 1


    print((count - error) / count)

