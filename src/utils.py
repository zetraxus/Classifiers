import csv

from sklearn.metrics import classification_report

from src.dataset import Dataset


def train(classifier, train_ds, column_info):
    classifier.train(train_ds, column_info)


def test(classifier, test_ds):
    true, predicted, i = [], [], 0
    for row in test_ds:
        i += 1
        if i == 30:
            break
        sample, gt = row[:-1], row[-1]
        predicted_class = classifier.predict(sample)
        true.append(gt), predicted.append(predicted_class)

    return calc_metrics(true, predicted)


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
def calc_metrics(true, predicted):
    return classification_report(true, predicted, zero_division=0)


def save_results(results_report, ds_name, classifier):
    print(ds_name, classifier)
    print(results_report)
    print("============")


def create_dataset(ds_path, ds_name, train_ratio, ds_info, buckets):
    with open(ds_path + ds_name + ".csv") as f:
        csv_input = csv.reader(f, delimiter=',')
        return Dataset(csv_input, train_ratio, ds_info[ds_name], buckets)


def split_dataset(dataset, classifier_name):
    if classifier_name in ["NaiveBayes", "SPRINT"]:
        return dataset.get_train_set(buckets=True), dataset.get_test_set(buckets=True)
    else:
        return dataset.get_train_set(buckets=False), dataset.get_test_set(buckets=False)
