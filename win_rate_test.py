import utility as ut
import pandas as pd






if __name__ == '__main__':

    dataframe1 = pd.read_excel('test.xlsx')
    utility = ut.Utility()
    tag = utility.make_list(data_frame=dataframe1, col_name='tag')
    label = utility.make_list(data_frame=dataframe1, col_name='label')
    print(label)
    print(dataframe1)
    good = 0
    bad = 0
    for i in range(len(label)):
        if label[i] == tag[i]:
            good += 1
        else:
            bad += 1

    print(good/(good + bad))