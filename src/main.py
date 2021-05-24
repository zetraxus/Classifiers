from src.algorithms.lcpc import LCPC
from src.algorithms.naive_bayes import NaiveBayes
from src.utils import create_dataset, train, test, save_results

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
        classifiers = [NaiveBayes(), LCPC()]

        for classifier in classifiers:
            if classifier.__class__.__name__ in ["NaiveBayes", "SPRINT"]:
                train_ds, test_ds = dataset.get_train_set(buckets=True), dataset.get_test_set(buckets=True)
            else:
                train_ds, test_ds = dataset.get_train_set(buckets=False), dataset.get_test_set(buckets=False)

            train(classifier, train_ds)
            metrics = test(classifier, test_ds)
            save_results(metrics, ds_name, classifier.__class__.__name__)
