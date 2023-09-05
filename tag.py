from scipy import signal


class Tag:
    def __init__(self, y_axis):
        self.y_axis = y_axis

    def tag_peak(self):
        """
        maximum value are tagged as False and minimum are tagged True
        """
        main_peaks, _ = signal.find_peaks(x=self.y_axis)
        negated_list = [-x for x in self.y_axis]
        main_valley, _ = signal.find_peaks(x=negated_list)
        tag_list = []
        if main_peaks[0] > main_valley[0]:
            non_extremum = False
        else:
            non_extremum = True
        for i in range(len(self.y_axis)):
            if i in main_peaks:
                tag_list.append(False)
                non_extremum = False
            elif i in main_valley:
                tag_list.append(True)
                non_extremum = True
            else:
                tag_list.append(non_extremum)
        return tag_list
