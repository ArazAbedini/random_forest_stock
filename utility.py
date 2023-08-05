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