import csv

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.algorithms.lcpc import LCPC
from src.algorithms.naive_bayes import NaiveBayes
from src.algorithms.sprint import SPRINT
from src.dataset import Dataset


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
def calc_metrics(true, predicted):
    prf = precision_recall_fscore_support(true, predicted, average="weighted", zero_division=0)
    acc = accuracy_score(true, predicted)
    return prf, acc


def save_results(metrics, ds_name, classifier_name):
    for i in range(len(metrics[0]) - 1):
        print(ds_name, classifier_name, metrics[0][i])
    print(ds_name, classifier_name, metrics[1])


if __name__ == "__main__":
    dataset_path = "../data/"

    # datasets_names = ["bank", "cmc", "diabetes", "occupancy", "skin", "wine"]
    datasets_names = ["skin"]
    train_ratio = 0.8

    for ds_name in datasets_names:
        with open(dataset_path + ds_name + ".csv") as f:
            csv_input = csv.reader(f, delimiter=',')

            # create dataset object
            dataset = Dataset(csv_input, train_ratio)
            train_ds = dataset.getTrainSet()
            test_ds = dataset.getTestSet()

        nb, lcpc, sprint = NaiveBayes(), LCPC(), SPRINT()
        # classifiers = [nb, lcpc, sprint]
        classifiers = [lcpc]
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
            save_results(metrics, ds_name, classifier.__class__.__name__)
