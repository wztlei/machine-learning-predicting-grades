import csv
from typing import List
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, hamming_loss, mean_squared_error
from sklearn.naive_bayes import GaussianNB


def cross_validation(students, classifier, n_splits=10, ):
    kf = KFold(n_splits=n_splits, shuffle=True)
    semester_grades = [s['semester_grades'] for s in students]
    final_grades = [s['final_grade'] for s in students]

    cumulative_stats = {
        'percent_correct': 0,
        'mean_squared_error': 0,
        'hamming_loss': 0
    }

    for train_indices, test_indices in kf.split(semester_grades):
        train_semester_grades = [semester_grades[i] for i in train_indices]
        train_final_grades = [final_grades[i] for i in train_indices]
        test_semester_grades = [semester_grades[i] for i in test_indices]
        test_final_grades = [final_grades[i] for i in test_indices]

        set_stats = classifier(
            train_semester_grades, train_final_grades,
            test_semester_grades, test_final_grades
        )

        for key in cumulative_stats.keys():
            cumulative_stats[key] += set_stats[key]

    avg_stats = {}

    for key in cumulative_stats.keys():
        avg_stats[key] = cumulative_stats[key] / n_splits

    print(avg_stats)


def k_nearest_neighbours_classifier(
        train_semester_grades, train_final_grades,
        test_semester_grades, test_final_grades, k=19):
    # Use the sklearn k-neighbours classifier to predict grades
    k_neighbours_classifier = KNeighborsClassifier(n_neighbors=k)
    k_neighbours_classifier.fit(train_semester_grades, train_final_grades)
    predictions = k_neighbours_classifier.predict(test_semester_grades)

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


def gaussian_naive_bayes_classifier(
        train_semester_grades, train_final_grades,
        test_semester_grades, test_final_grades):
    gaussian_nb_classifier = GaussianNB()
    gaussian_nb_classifier.fit(train_semester_grades, train_final_grades)
    predictions = gaussian_nb_classifier.predict(test_semester_grades)

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


def parse_dataset(
        filename: str,
        print_file: bool = False,
        ignore_imputation=True
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
            print("len(lines) =", len(raw_data))

            for line in raw_data:
                print(line)

        raw_data_without_headers = raw_data[1:]

        if ignore_imputation:
            grades_end_index = -5
        else:
            grades_end_index = -2

        # Organize the data into a list of dicts which each represent a student
        students = [{
            'id': student[0],
            'year': student[1],
            'semester_grades': [float(grade) for grade in
                                student[2:grades_end_index]],
            'predicted_grade': int(student[-2]),
            'final_grade': int(student[-1])
        } for student in raw_data_without_headers]

        return students


def main():
    # Retrieve the raw data from the csv file
    students = parse_dataset('model_input_data.csv', print_file=False)
    cross_validation(students, k_nearest_neighbours_classifier,
                     n_splits=len(students))
    cross_validation(students, gaussian_naive_bayes_classifier,
                     n_splits=len(students))


if __name__ == '__main__':
    main()
