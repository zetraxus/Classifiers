from src.algorithm import Algorithm


class NaiveBayes(Algorithm):
    def __init__(self):
        super().__init__()
        self.histogram_classes = None
        self.histogram_all = None
        self.class_distr = dict()

    def train(self, ds):
        self.histogram_classes = self.__calc_attr_values_histogram(ds)
        self.histogram_all = self.__calc_all_attr_values_histogram()

    def predict(self, sample):
        probabilities = list()

        for cl in self.histogram_classes:
            prob = self.__calc_prob(cl, sample)
            probabilities.append((cl, prob))

        probabilities.sort(key=lambda tup: tup[1], reverse=True)
        return probabilities[0][0]

    def __calc_attr_values_histogram(self, ds):
        histogram_cl = dict()
        for row in ds:
            row_len, cl = len(row), row[-1]
            if cl not in histogram_cl:
                histogram_cl[cl] = list()
                for _ in range(len(row) - 1):
                    histogram_cl[cl].append(dict())

            for i in range(len(row) - 1):
                attr = row[i]
                if attr not in histogram_cl[cl][i]:
                    histogram_cl[cl][i][attr] = 1
                else:
                    histogram_cl[cl][i][attr] += 1

            if cl not in self.class_distr:
                self.class_distr[cl] = 1
            else:
                self.class_distr[cl] += 1

        return histogram_cl

    def __calc_all_attr_values_histogram(self):
        histogram_all, row_len = list(), 0
        for i in range(row_len - 1):
            histogram_all.append(dict())
            for v in self.histogram_classes.values():
                for attr_val, cnt in v[i].items():
                    if attr_val in histogram_all[i]:
                        histogram_all[i][attr_val] += cnt
                    else:
                        histogram_all[i][attr_val] = cnt

        return histogram_all

    def __calc_prob(self, cl, sample):
        return 0