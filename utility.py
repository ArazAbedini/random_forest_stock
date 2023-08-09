class Utility:
    def __init__(self):
        pass

    @staticmethod
    def make_list(data_frame, col_name):
        output = []
        for index, row in data_frame.iterrows():
            output.append(row[col_name])
        return output

    @staticmethod
    def delete_char(data_frame, col_name):
        for index, row in data_frame.iterrows():
            row[col_name] = row[col_name][:-1]
        return data_frame

    @staticmethod
    def remove_date(df, utility):
        date = utility.make_list(df, 'date')
        df.drop('date', inplace=True, axis=1)

    @staticmethod
    def title_label(data_frame):
        data_frame = data_frame.iloc[1:]
        col_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']
        data_frame.columns = col_names
        return data_frame


    @staticmethod
    def rist_to_reward_ratio(x1, x2):
        return (0.95 * x1) / (x2 - x1)

    @staticmethod
    def tagg(list):
        tag_list = []
        for i in range(len(list)):
            if i == 0:
                tag_list.append(True)
                temp = list[i]
            else:
                try:
                    rr_ratio = ((0.95 * temp) / (list[i] - temp))
                    if rr_ratio >= 1.5:
                        tag_list.append(False)
                    else:
                        tag_list.append(True)
                        temp = list[i]
                except:
                    tag_list.append(False)
                    temp = list[i]

        return tag_list
