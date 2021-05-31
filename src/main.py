import time

from src.algorithms.lcpc import LCPC
from src.algorithms.naive_bayes import NaiveBayes
from src.algorithms.sprint import SPRINT
from src.utils import create_dataset, train, test, save_results, split_dataset

if __name__ == "__main__":
    dataset_path, train_ratio, buckets = "../data/", 0.9, 10
    datasets_info = {
        "occupancy": ['nc'] * 5,
        "skin": ['nc'] * 3,
        "cmc": ['nd'] * 9,
        "diabetes": ['nd', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc'],
        "wine": ['nc'] * 11,
        "bank": ['nc', 'e', 'e', 'e', 'e', 'nc', 'e', 'e', 'e', 'nc', 'nc', 'nc', 'e'],
    }

    for ds_name in datasets_info.keys():
        dataset = create_dataset(dataset_path, ds_name, train_ratio, datasets_info, buckets)

        classifiers = [
            LCPC(),
            NaiveBayes(0),
            NaiveBayes(0.1),
            NaiveBayes(0.5),
            NaiveBayes(1),
            SPRINT()
        ]

        for classifier in classifiers:
            train_ds, test_ds = split_dataset(dataset, classifier.__class__.__name__)
            start = time.time()
            train(classifier, train_ds, column_info=dataset.get_info())
            results_report = test(classifier, test_ds)
            end = time.time()
            save_results(results_report, ds_name, classifier, end - start)
