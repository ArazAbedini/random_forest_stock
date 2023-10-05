


def risk_reward_compute(test_list,price_list, date_list):
    buy = None
    incorrect_counter = 0
    correct_counter = 0
    for i in range(len(test_list)):
        if test_list[i] == 'S/B':
            if buy != None:
                risk_reward = (0.95 * buy) / abs(price_list[i] - buy)
                if risk_reward < 1.5:
                    print(buy, price_list[i], date_list[i])
                    incorrect_counter += 1
                else:
                    correct_counter += 1
            buy = price_list[i]
        elif test_list[i] == 'B':
            buy = price_list[i]
        elif test_list[i] == 'S':
            if buy != None:
                risk_reward = (0.95 * buy) / abs(price_list[i] - buy)
                if risk_reward < 1.5:
                    print(buy, price_list, date_list[i])
                    incorrect_counter += 1
                else:
                    correct_counter += 1
    print(correct_counter / (correct_counter + incorrect_counter))
    return True
