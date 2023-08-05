class Tag:
    def __init__(self, historical_changes):
        self.historical_changes = historical_changes

    def positive_number(self):
        if all(i < j for i, j in zip(self.historical_changes, self.historical_changes[1:])):
            res = True
        elif all(i > j for i, j in zip(self.historical_changes, self.historical_changes[1:])):
            res = False if self.historical_changes[2] < 0.5 else True
        else:
            res = True
        return res

    def negative_number(self):
        if all(i > j for i, j in zip(self.historical_changes, self.historical_changes[1:])):
            res = False
        elif all(i < j for i, j in zip(self.historical_changes, self.historical_changes[1:])):
            res = True if abs(self.historical_changes[2]) < 0.5 else False
        else:
            res = False
        return res

    def complicated_number(self):
        if self.historical_changes[0] > 0:
            if self.historical_changes[1] < 0:
                res = True if self.historical_changes[2] > 0 else False
            else:
                res = False if abs(self.historical_changes[1]) < 0.5 else True
        else:
            if self.historical_changes[1] > 0:
                res = True if self.historical_changes[2] > 0 else False
            else:
                res = True if abs(self.historical_changes[1]) < 0.5 else False
        return res

    def tag_point(self):
        res = all(i > 0 and j > 0 for i, j in
                  zip(self.historical_changes,
                      self.historical_changes[1:]))  # res = True if it means that we have to buy
        if res:
            res = self.positive_number()
        elif all(i < 0 and j < 0 for i, j in zip(self.historical_changes, self.historical_changes[1:])):
            res = self.negative_number()
        else:
            res = self.complicated_number()
        return res
