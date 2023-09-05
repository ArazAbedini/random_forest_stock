

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



# if __name__ == '__main__':
#     price_list = [10, 12, 11, 14, 9, 15, 16, 17, 8]
#     decision_list = ['B', 'S', 'B', 'S/B', 'S', 'N', 'S', 'B', 'S']
#     process_win_rate = WinRate(decision_list=decision_list, price_list=price_list)
#     process_win_rate.calculate_win_rate()
#     wind_rate = process_win_rate.win_rate()
#     print(wind_rate)