import csv
import itertools
import json
import warnings
from typing import List

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, \
    ComplementNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


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
        round_grades: True if the grades are to be rounded, and False otherwise.

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
    # semester_grades = preprocessing.scale(
    #     [s['semester_grades'] for s in students]
    # )
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


def optimize_logistic_regression(students):
    solver_list = ['newton-cg', 'liblinear']
    penalty_list = ['l2']
    list_of_args = [solver_list, penalty_list]

    for solver, penalty in itertools.product(*list_of_args):
        stats = classify_and_cross_validate(students, LogisticRegression(
            solver=solver,
            penalty=penalty
        ))
        print(stats)

    # The optimal solver determined empirically is newton-cg
    return LogisticRegression(
        solver='newton-cg',
        penalty='l2'
    )


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


def optimize_adaboost_classifier(students):
    adaboost_classifier = AdaBoostClassifier(
        n_estimators=150,
        base_estimator=DecisionTreeClassifier(max_depth=1)
    )
    stats = classify_and_cross_validate(students, adaboost_classifier)
    # TODO: Find a way to optimize Adaboost to increase the algorithm's speed
    print(stats)


def optimize_random_forest_classifier(students):
    n_estimators_list = list(range(90, 250, 30))
    max_features_list = list(range(1, 10))

    # criterion_list = ['gini', 'entropy']
    # max_depth_list = [None, *list(range(5, 9))]
    # min_samples_split_list = [2, 4]
    # min_samples_leaf_list = [1, 3]
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
    #   "percent_correct": 64.52380952380952,
    #   "mean_squared_error": 0.5504761904761906,
    #   "hamming_loss": 0.3547619047619048
    # }
    # {
    #   "n_estimators": 120,
    #   "max_features": 4
    #   "max_depth': 5,
    #   "min_samples_split": 3,
    #   "min_samples_leaf": 3
    # }


def optimize_svc(students):
    # noinspection PyPep8Naming
    C_list = [0.0001, 0.0005, 0.001, 0.1]
    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

    list_of_args = [C_list, kernel_list]

    count = 1
    best_stats = None
    best_args = None

    for C, kernel in itertools.product(*list_of_args):
        stats = classify_and_cross_validate(
            students, SVC(
                # C=C,
                kernel=kernel
            )
        )

        if best_stats is None or \
                stats['percent_correct'] > best_stats['percent_correct']:
            best_stats = stats
            best_args = {
                'C': C,
                'kernel': kernel
            }

        print(count)
        print(json.dumps(best_stats, indent=2))
        print(json.dumps(best_args, indent=2))
        count += 1

    print(json.dumps(best_stats, indent=2))
    print(json.dumps(best_args, indent=2))

    # TODO: Find a way to optimize SVC to increase the algorithm's speed

    return SVC(
        C=best_args['C'],
        kernel=best_args['kernel']
    )

    # Sample Result:
    # {
    #   "percent_correct": 49.828042328042315,
    #   "mean_squared_error": 0.7564814814814815,
    #   "hamming_loss": 0.5017195767195768
    # }
    # {
    #   "C": 0.0001,
    #   "kernel": "linear"
    # }


def optimize_linear_svc(students):
    penalty_list = ['l2']
    # noinspection PyPep8Naming
    C_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    list_of_args = [penalty_list, C_list]

    count = 1
    best_stats = None
    best_args = None

    for penalty, C in itertools.product(*list_of_args):
        stats = classify_and_cross_validate(
            students, LinearSVC(
                penalty=penalty,
                C=C
            )
        )

        if best_stats is None or \
                stats['percent_correct'] > best_stats['percent_correct']:
            best_stats = stats
            best_args = {
                'penalty': penalty,
                'C': C,
            }

        print(count)
        count += 1

    print(json.dumps(best_stats, indent=2))
    print(json.dumps(best_args, indent=2))

    return LinearSVC(
        penalty=best_args['penalty'],
        C=best_args['C']
    )

    # Sample output
    # {
    #   "percent_correct": 47.31481481481482,
    #   "mean_squared_error": 0.7947089947089947,
    #   "hamming_loss": 0.5268518518518518
    # }
    # {
    #   "penalty": "l2",
    #   "C": 0.0005
    # }


def optimize_multi_layer_perceptron(students):
    activation_list = ['identity', 'logistic', 'tanh', 'relu']
    solver_list = ['lbfgs', 'sgd', 'adam']

    list_of_args = [activation_list, solver_list]

    count = 1
    best_stats = None
    best_args = None

    for activation, solver in itertools.product(*list_of_args):
        stats = classify_and_cross_validate(
            students, MLPClassifier(
                activation=activation,
                solver=solver
            )
        )

        if best_stats is None or \
                stats['percent_correct'] > best_stats['percent_correct']:
            best_stats = stats
            best_args = {
                'activation': activation,
                'solver': solver,
            }

        print(count)
        count += 1

    print(json.dumps(best_stats, indent=2))
    print(json.dumps(best_args, indent=2))

    return MLPClassifier(
        activation=best_args['activation'],
        solver=best_args['solver']
    )

    # Sample Output:
    # {
    #   "percent_correct": 58.32275132275132,
    #   "mean_squared_error": 0.5546296296296298,
    #   "hamming_loss": 0.41677248677248686
    # }
    # {
    #   "activation": "logistic",
    #   "solver": "adam"
    # }


def optimize_multi_layer_perceptron_layers(students):
    hidden_layer_sizes_list = [(5, 2), (15,), (100,), (100, 10),
                               (130, 10), (50, 10), (10, 10)]

    count = 1
    best_stats = None
    best_args = None

    for hidden_layer_sizes in hidden_layer_sizes_list:
        stats = classify_and_cross_validate(
            students, MLPClassifier(
                activation='logistic',
                solver='adam',
                hidden_layer_sizes=hidden_layer_sizes,
            )
        )

        if best_stats is None or \
                stats['percent_correct'] > best_stats['percent_correct']:
            best_stats = stats
            best_args = {
                'hidden_layer_sizes': hidden_layer_sizes,
            }

        print(count)
        count += 1

    print(json.dumps(best_stats, indent=2))
    print(json.dumps(best_args, indent=2))

    return MLPClassifier(
        hidden_layer_sizes=best_args['hidden_layer_sizes'],
    )

    # Sample Output:
    # {
    #   "percent_correct": 57.05555555555556,
    #   "mean_squared_error": 0.665952380952381,
    #   "hamming_loss": 0.42944444444444444
    # }
    # {
    #   "hidden_layer_sizes": [
    #     100
    #   ]
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
    # optimize_decision_tree_classifier(students)
    # optimize_random_forest_classifier(students)
    # optimize_logistic_regression(students)
    # optimize_adaboost_classifier(students)
    # optimize_svc(students)
    # optimize_multi_layer_perceptron(students)
    optimize_multi_layer_perceptron_layers(students)


if __name__ == '__main__':
    main()
