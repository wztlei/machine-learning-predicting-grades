import csv
import itertools
import json
import warnings
from typing import List

from sklearn.metrics import accuracy_score, hamming_loss, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, \
    ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def parse_dataset(
        filename: str,
        print_file: bool = False,
        ignore_imputation=True,
        round_grades=False
) -> List:
    """Reads the csv file containing the data for this project.

    Args:
        filename: A string denoting the name of the file storing the data.
        print_file: True if the file is to be printed, and False otherwise.
        ignore_imputation: True if the columns containing whether a value was
            imputed are to be ignored, and False otherwise.

    Returns:
        A list of all of the students with data for each person.
    """

    # Use a with/as block, which abstracts away opening and closing a file
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        raw_data = list(csv_reader)

        # Print the file contents if necessary
        if print_file:
            print('len(lines) =', len(raw_data))

            for line in raw_data:
                print(line)

        raw_data_without_headers = raw_data[1:]

        if ignore_imputation:
            grades_end_index = -5
        else:
            grades_end_index = -2

        # Organize the data into a list of dicts which each represent a student
        if round_grades:
            students = [{
                'id': student[0],
                'year': student[1],
                'semester_grades': [float(grade) for grade in
                                    student[2:grades_end_index]],
                'predicted_grade': int(student[-2]),
                'final_grade': int(student[-1])
            } for student in raw_data_without_headers]
        else:
            students = [{
                'id': student[0],
                'year': student[1],
                'semester_grades': [round(float(grade)) for grade in
                                    student[2:grades_end_index]],
                'predicted_grade': int(student[-2]),
                'final_grade': int(student[-1])
            } for student in raw_data_without_headers]

        return students


def classify_and_cross_validate(students, sklearn_classifier, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True)
    semester_grades = [s['semester_grades'] for s in students]
    final_grades = [s['final_grade'] for s in students]

    cumulative_stats = {
        'percent_correct': 0,
        'mean_squared_error': 0,
        'hamming_loss': 0
    }

    for train_indices, test_indices in kf.split(semester_grades, final_grades):
        train_semester_grades = [semester_grades[i] for i in train_indices]
        train_final_grades = [final_grades[i] for i in train_indices]
        test_semester_grades = [semester_grades[i] for i in test_indices]
        test_final_grades = [final_grades[i] for i in test_indices]

        set_stats = classify_with_sklearn(
            train_semester_grades, train_final_grades,
            test_semester_grades, test_final_grades,
            sklearn_classifier
        )

        for key in cumulative_stats.keys():
            cumulative_stats[key] += set_stats[key]

    avg_stats = {}

    for key in cumulative_stats.keys():
        avg_stats[key] = cumulative_stats[key] / n_splits

    # print(avg_stats)
    return avg_stats


def classify_with_sklearn(train_semester_grades, train_final_grades,
                          test_semester_grades, test_final_grades,
                          sklearn_classifier):
    # Use the sklearn k-neighbours classifier to predict grades
    sklearn_classifier.fit(train_semester_grades, train_final_grades)
    predictions = sklearn_classifier.predict(test_semester_grades)

    # Calculate the results of the model
    pc = accuracy_score(test_final_grades, predictions) * 100
    mse = mean_squared_error(test_final_grades, predictions)
    hl = hamming_loss(test_final_grades, predictions)

    # Return the results
    return {
        'percent_correct': pc,
        'mean_squared_error': mse,
        'hamming_loss': hl
    }


def optimize_k_neighbors_classifier(students):
    n_neighbors_list = list(range(1, 31))
    weights_list = ['uniform', 'distance']
    # algorithm_list = ['auto', 'ball_tree', 'kd_tree', 'brute']
    p_list = list(range(1, 5))
    metric_list = ['minkowski', 'chebyshev', 'canberra', 'braycurtis']
    list_of_args = [n_neighbors_list, weights_list, p_list, metric_list]

    count = 1
    best_stats = None
    best_args = None

    for n_neighbors, weights, p, metric in itertools.product(*list_of_args):
        if metric != 'minkowski' and p != p_list[0]:
            continue

        stats = classify_and_cross_validate(
            students, KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                p=p,
                metric=metric
            ), n_splits=10)

        if best_stats is None \
                or stats['percent_correct'] > best_stats['percent_correct']:
            best_stats = stats
            best_args = {
                'n_neighbors': n_neighbors,
                'weights': weights,
                'p': p,
                'metric': metric
            }

        if count % 10 == 0:
            print(count)
        count += 1

    # Print the stats and args for the best model
    print(json.dumps(best_stats, indent=2))
    print(json.dumps(best_args, indent=2))

    return KNeighborsClassifier(
        n_neighbors=best_args['n_neighbors'],
        weights=best_args['weights'],
        p=best_args['p'],
        metric=best_args['metric']
    )

    # Sample Results:
    # {
    #   "percent_correct": 60.91269841269841,
    #   "mean_squared_error": 0.5623015873015873,
    #   "hamming_loss": 0.3908730158730159
    # }
    # {
    #   "n_neighbors": 15,
    #   "weights": "uniform",
    #   "p": 1,
    #   "metric": "minkowski"
    # }


def optimize_radius_neighbors_classifier(students):
    radius_list = list(range(80, 160, 10))
    weights_list = ['uniform', 'distance']
    # algorithm_list = ['auto', 'ball_tree', 'kd_tree', 'brute']
    p_list = list(range(1, 5))
    metric_list = ['minkowski', 'chebyshev', 'canberra', 'braycurtis']
    list_of_args = [radius_list, weights_list, p_list, metric_list]

    count = 1
    best_stats = None
    best_args = None

    for radius, weights, p, metric in itertools.product(*list_of_args):
        if metric != 'minkowski' and p != p_list[0]:
            continue

        stats = classify_and_cross_validate(
            students, RadiusNeighborsClassifier(
                radius=radius,
                weights=weights,
                p=p,
                metric=metric,
                outlier_label=1
            ), n_splits=10)

        if best_stats is None \
                or stats['percent_correct'] > best_stats['percent_correct']:
            best_stats = stats
            best_args = {
                'radius': radius,
                'weights': weights,
                'p': p,
                'metric': metric
            }

        if count % 10 == 0:
            print(count)
        count += 1

    print(json.dumps(best_stats, indent=2))
    print(json.dumps(best_args, indent=2))

    # Print the stats and args for the best model
    return RadiusNeighborsClassifier(
        radius=best_args['radius'],
        weights=best_args['weights'],
        p=best_args['p'],
        metric=best_args['metric'],
        outlier_label=1
    )

    # Sample Result:
    # {
    #   "percent_correct": 53.79629629629629,
    #   "mean_squared_error": 0.7914021164021163,
    #   "hamming_loss": 0.4620370370370369
    # }
    # {
    #   "radius": 100,
    #   "weights": "distance",
    #   "p": 1,
    #   "metric": "minkowski"
    # }


def optimize_naive_bayes(students):
    naive_bayes_models = {
        'BernoulliNB': BernoulliNB(),
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'ComplementNB': ComplementNB()
    }

    best_stats = None
    best_model_name = None

    for model_name in naive_bayes_models.keys():
        stats = classify_and_cross_validate(
            students, naive_bayes_models[model_name], n_splits=10
        )

        if best_stats is None \
                or stats['percent_correct'] > best_stats['percent_correct']:
            best_stats = stats
            best_model_name = model_name

    print(json.dumps(best_stats, indent=2))
    print(best_model_name)

    return naive_bayes_models[best_model_name]


def optimize_decision_tree_classifier(students):
    criterion_list = ["gini"]
    splitter_list = ["best"]
    max_depth_list = [None, *range(1, 8)]
    min_samples_split_list = [*range(2, 9)]
    min_samples_leaf_list = [*range(1, 8)]

    list_of_args = [criterion_list, splitter_list, max_depth_list,
                    min_samples_split_list, min_samples_leaf_list]

    count = 1
    best_stats = None
    best_args = None

    for criterion, splitter, max_depth, min_samples_split, min_samples_leaf \
            in itertools.product(*list_of_args):
        stats = classify_and_cross_validate(
            students, DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
        )

        if best_stats is None or \
                stats['percent_correct'] > best_stats['percent_correct']:
            best_stats = stats
            best_args = {
                'criterion': criterion,
                'splitter': splitter,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }

        if count % 10 == 0:
            print(count)

        count += 1

    print(json.dumps(best_stats, indent=2))
    print(json.dumps(best_args, indent=2))

    return DecisionTreeClassifier(
        criterion=best_args['criterion'],
        splitter=best_args['splitter'],
        max_depth=best_args['max_depth'],
        min_samples_split=best_args['min_samples_split'],
        min_samples_leaf=best_args['min_samples_leaf']
    )

    # Sample Output
    # {
    #   "percent_correct": 59.86772486772486,
    #   "mean_squared_error": 0.6357142857142857,
    #   "hamming_loss": 0.40132275132275125
    # }
    # {
    #   "criterion": "gini",
    #   "splitter": "best",
    #   "max_depth": 3,
    #   "min_samples_split": 6,
    #   "min_samples_leaf": 4
    # }


def optimize_random_forest_classifier(students):
    n_estimators_list = list(range(90, 250, 30))
    max_features_list = list(range(1, 10))

    # criterion_list = ['gini', 'entropy']
    # max_depth_list = [None, *list(range(5, 8))]
    # min_samples_split_list = [2, 3]
    # min_samples_leaf_list = [1, 2]
    list_of_args = [n_estimators_list, max_features_list]
    # list_of_args = [n_estimators_list, criterion_list, max_depth_list,
    #                 min_samples_split_list, min_samples_leaf_list]

    count = 1
    best_stats = None
    best_args = None

    # for n_estimators, criterion, max_depth, min_samples_split,
    # min_samples_leaf \ in itertools.product(*list_of_args):
    for n_estimators, max_features in itertools.product(*list_of_args):
        stats = classify_and_cross_validate(
            students, RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features
                # max_depth=max_depth,
                # min_samples_split=min_samples_split,
                # min_samples_leaf=min_samples_leaf
            )
        )

        if best_stats is None or \
                stats['percent_correct'] > best_stats['percent_correct']:
            best_stats = stats
            best_args = {
                'n_estimators': n_estimators,
                'max_features': max_features
                # 'max_depth': max_depth,
                # 'min_samples_split': min_samples_split,
                # 'min_samples_leaf': min_samples_leaf
            }

        if count % 5 == 0:
            print(count, n_estimators, max_features)
            print(best_stats)
            print(best_args)
            print('')

        count += 1

    print(json.dumps(best_stats, indent=2))
    print(json.dumps(best_args, indent=2))

    return RandomForestClassifier(
        n_estimators=best_args['n_estimators'],
        max_features=best_args['max_features']
        # max_depth=best_args['max_depth'],
        # min_samples_split=best_args['min_samples_split'],
        # min_samples_leaf=best_args['min_samples_leaf']
    )
    
    # Sample Result:
    # {
    #   "percent_correct": 58.12169312169311,
    #   "mean_squared_error": 0.601984126984127,
    #   "hamming_loss": 0.4187830687830688
    # }
    # {
    #   "n_estimators": 120,
    #   "max_features": 4
    # }


def main():
    # Retrieve the raw data from the csv file
    warnings.simplefilter(action='ignore', category=FutureWarning)
    students = parse_dataset(
        'model_input_data.csv', print_file=False, round_grades=False
    )

    # optimize_k_neighbors_classifier(students)
    # optimize_radius_neighbors_classifier(students)
    # optimize_naive_bayes(students)
    best = optimize_decision_tree_classifier(students)


    for x in range(0, 20):
        stats = classify_and_cross_validate(
            students, best
        )
        print(stats)

    # optimize_random_forest_classifier(students)


if __name__ == '__main__':
    main()
