def limit_fuction(test_list, price_list):
    buy = None
    for i in range(len(test_list)):
        if test_list[i] == 'B':
            if buy != None:
                if price_list[i] > 1.95 * buy or price_list[i] < 0.95 * buy:
                    test_list[i] = 'S'
            buy = price_list[i]
        elif test_list[i] == 'S':
            buy = None
        elif test_list[i] == 'S/B':
            if buy != None:
                buy = price_list[i]
            else:
                buy = None
        else:
            if buy != None:
                if price_list[i] >= 1.95 * buy or price_list[i] < 0.95 * buy:
                    test_list[i] = 'S'
                    buy = price_list[i]
    return test_list