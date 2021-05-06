import csv

from src.algorithms import NaiveBayes, LCCP, SPRINT
from src.dataset import Dataset


def calc_metrics(true, predicted):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    # TODO
    return []


def save_results(metrics, ds_name, classifier_name):
    # TODO
    pass


if __name__ == "__main__":
    dataset_path = "../data/"

    datasets_names = ["bank", "cmc", "diabetes", "occupancy", "skin", "wine"]
    train_ratio = 0.8

    for ds_name in datasets_names:
        with open(dataset_path + ds_name + ".csv") as f:
            csv_input = csv.reader(f, delimiter=',')

            # create dataset object
            dataset = Dataset(csv_input, train_ratio)
            train_ds = dataset.getTrainSet()
            test_ds = dataset.getTestSet()

        nb, lccp, sprint = NaiveBayes(), LCCP(), SPRINT()
        classifiers = [nb, lccp, sprint]
        for classifier in classifiers:
            # train
            classifier.train(train_ds)

            # test
            true, predicted = [], []
            for row in test_ds:
                sample, gt = row[:-1], row[-1]
                predicted_class = classifier.predict(sample)
                true.append(gt), predicted.append(predicted_class)

            # calc metrics
            metrics = calc_metrics(true, predicted)

            # save results
            save_results(metrics, ds_name, classifier.__name__)
