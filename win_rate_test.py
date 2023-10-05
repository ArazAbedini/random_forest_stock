import utility as ut
import pandas as pd


def compute_win_rate(train, test):
    good = 0
    bad = 0
    for i in range(len(train)):
        if train[i] == test[i]:
            good += 1
        else:
            bad += 1
    return good/(good + bad)


