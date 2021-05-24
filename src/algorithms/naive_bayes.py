from src.algorithm import Algorithm


class NaiveBayes(Algorithm):
    def __init__(self, laplace_smoothing=0):
        super().__init__()
        self.hist_cl = None
        self.hist_all = None
        self.train_ds_size = None
        self.attr_cnt = None
        self.smth = laplace_smoothing
        self.class_distr = dict()

    def train(self, ds):
        self.hist_cl = self.__calc_attr_values_histogram(ds)
        self.hist_all = self.__calc_all_attr_values_histogram()
        self.train_ds_size = len(ds)

    def predict(self, sample):
        probabilities = list()

        for cl in self.hist_cl:
            prob = self.__calc_prob(cl, sample)
            probabilities.append((cl, prob))

        probabilities.sort(key=lambda tup: tup[1], reverse=True)

        if probabilities[0][1] == probabilities[-1][1]:
            return self.__most_popular_class()

        return probabilities[0][0]

    def __calc_attr_values_histogram(self, ds):
        histogram_cl = dict()
        for row in ds:
            self.attr_cnt, cl = len(row), row[-1]
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
        histogram_all = list()
        for i in range(self.attr_cnt - 1):
            histogram_all.append(dict())
            for v in self.hist_cl.values():
                for attr_val, cnt in v[i].items():
                    if attr_val in histogram_all[i]:
                        histogram_all[i][attr_val] += cnt
                    else:
                        histogram_all[i][attr_val] = cnt

        return histogram_all

    def __calc_prob(self, cl, sample):
        prob = 1
        for attr_id in range(len(sample)):
            if sample[attr_id] in self.hist_cl[cl][attr_id]:
                prob *= (self.hist_cl[cl][attr_id][sample[attr_id]] + self.smth) / (self.class_distr[cl] + self.smth)
                prob /= self.hist_all[attr_id][sample[attr_id]]
            else:
                prob = 0
        prob *= self.class_distr[cl] / self.train_ds_size

        return prob

    def __most_popular_class(self):
        cl, cnt = None, -1
        for k, v in self.class_distr.items():
            if v > cnt:
                cl, cnt = k, v
        return cl

    def __str__(self):
        return self.__class__.__name__ + " " + str(self.smth)
