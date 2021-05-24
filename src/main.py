from src.algorithms.lcpc import LCPC
from src.algorithms.naive_bayes import NaiveBayes
from src.utils import create_dataset, train, test, save_results, split_dataset

if __name__ == "__main__":
    dataset_path, train_ratio, buckets = "../data/", 0.8, 10
    datasets_info = {
        "skin": ['n'] * 3,
        "bank": ['n', 'e', 'e', 'e', 'e', 'n', 'e', 'e', 'e', 'n', 'n', 'n', 'e'],
        "cmc": ['n'] * 9,
        "diabetes": ['n'] * 8,
        "occupancy": ['n'] * 5,
        "wine": ['n'] * 11
    }

    for ds_name in datasets_info.keys():
        dataset = create_dataset(dataset_path, ds_name, train_ratio, datasets_info, buckets)
        # classifiers = [NaiveBayes(), LCPC(), SPRINT()]
        classifiers = [NaiveBayes(0),
                       NaiveBayes(0.1),
                       NaiveBayes(0.5),
                       NaiveBayes(1),
                       LCPC()]

        for classifier in classifiers:
            train_ds, test_ds = split_dataset(dataset, classifier.__class__.__name__)
            train(classifier, train_ds)
            results_report = test(classifier, test_ds)
            save_results(results_report, ds_name, classifier)
