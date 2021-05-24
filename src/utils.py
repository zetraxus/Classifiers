import csv

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.dataset import Dataset


def train(classifier, train_ds):
    classifier.train(train_ds)


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
    prf = precision_recall_fscore_support(true, predicted, zero_division=0)
    acc = accuracy_score(true, predicted)
    return prf, acc


def save_results(metrics, ds_name, classifier_name):
    for i in range(len(metrics[0]) - 1):
        print(ds_name, classifier_name, metrics[0][i])
    print(ds_name, classifier_name, metrics[1])


def create_dataset(ds_path, ds_name, train_ratio, ds_info, buckets):
    with open(ds_path + ds_name + ".csv") as f:
        csv_input = csv.reader(f, delimiter=',')
        return Dataset(csv_input, train_ratio, ds_info[ds_name], buckets)


def split_dataset(dataset, classifier_name):
    if classifier_name in ["NaiveBayes", "SPRINT"]:
        return dataset.get_train_set(buckets=True), dataset.get_test_set(buckets=True)
    else:
        return dataset.get_train_set(buckets=False), dataset.get_test_set(buckets=False)
