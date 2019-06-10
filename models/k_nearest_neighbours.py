import csv
import math
import matplotlib.pylab as plt
from typing import List, Dict, Callable


def load_dataset(filename: str, print_file: bool = False) -> List[List[str]]:
    """Reads the csv file containing the data for this project.

    Args:
        filename: A string denoting the name of the file storing the data.
        print_file: True if the file is to be printed, and False otherwise.

    Returns:
        A list of all of the lines cf text contained within the data file.
    """

    # Use a with/as block, which abstracts away opening and closing a file
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lines = list(csv_reader)

        # Print the file contents if necessary
        if print_file:
            print('len(lines) =', len(lines))

            for line in lines:
                print(line)

        # Return the file contents as a 2D list of strings
        return lines


def minkowski_distance(point_1: List[float], point_2: List[float],
                       p: int) -> float:
    """Calculates the Minkowski distance between two points.

    From Wikipedia: The Minkowski distance of order p between two points
    X = (x1, ... , xn) and Y = (y1, ... , yn) is defined as
    D(X, Y) = (sum_from_1_to_n( |xi - yi|^p )) ^ (1/p).

        Args:
            point_1: The coordinates for the 1st point as a list.
            point_2: The coordinates for the 2nd point as a list.
            p: The p-value for the Minkowski distance function.

        Returns:
            The Minkowski distance between point_1 and point_2.
        """

    sum_of_squares = 0

    for (val_1, val_2) in zip(point_1, point_2):
        sum_of_squares += math.pow(math.fabs(val_1 - val_2), p)

    return math.pow(sum_of_squares, 1/p)


def manhattan_distance(point_1: List[float], point_2: List[float]) -> float:
    """Calculates the Manhattan distance between two points.

    Args:
        point_1: The coordinates for the 1st point as a list.
        point_2: The coordinates for the 2nd point as a list.

    Returns:
        The Manhattan distance between point_1 and point_2.
    """
    return minkowski_distance(point_1, point_2, 1)


def euclidean_distance(point_1: List[float], point_2: List[float]) -> float:
    """Calculates the Euclidean distance between two points.

    Args:
        point_1: The coordinates for the 1st point as a list.
        point_2: The coordinates for the 2nd point as a list.

    Returns:
        The Euclidean distance between point_1 and point_2.
    """
    return minkowski_distance(point_1, point_2, 2)


def insert_nearest_neighbours(nearest_neighbours: List[Dict],
                              k: int, new_neighbour: Dict) -> None:
    """Updates the list of nearest neighbours by inserting a new point

    If nearest_neighbours has less than k items, then new_neighbour is inserted
    into nearest_neighbours. Otherwise, new_neighbour is inserted into the list
    and the item in nearest_neighbours with the greatest distance is removed.
    nearest_neighbours always has the property that it is sorted with the
    items increasing in distance from the data point currently examined.

    Args:
        nearest_neighbours: The pre-existing sorted list of neighbours that are
            closest to the data point currently being classified.
        k: The maximum size of nearest_neighbours.
        new_neighbour: The new neighbouring data point to insert.
    """
    is_new_neighbour_inserted = False

    # Iterate through all possible positions to insert in nearest_neighbours
    for i in range(len(nearest_neighbours)):
        # Insert in the appropriate place to preserve the sorted property
        if new_neighbour['distance'] < nearest_neighbours[i]['distance']:
            nearest_neighbours.insert(i, new_neighbour)
            is_new_neighbour_inserted = True
            break

    # Append to the end of the list if necessary
    if len(nearest_neighbours) < k and not is_new_neighbour_inserted:
        nearest_neighbours.append(new_neighbour)
    # Remove the item in the list with the largest distance if necessary
    elif len(nearest_neighbours) > k:
        nearest_neighbours.pop()


def k_nearest_neighbours_from_scratch (
        students: List[Dict], k: int,
        distance_function: Callable = euclidean_distance) -> Dict:
    """Predicts students' final grades with the k-nearest neighbours algorithm.

    Args:
        students: A list of dicts storing the data for each student,
            including assessment marks and final grades.
        k: The number of nearest neighbours to examine in the algorithm.
        distance_function: The distance function to use for the algorithm.
            Defaults to the Euclidean distance.
    Returns:
        A dict storing the results of the predictive algorithm.
    """
    num_correct = 0
    sum_squared_errors = 0

    # Iterate through every student to classify each one
    for student in students:
        other_students = [s for s in students if s != student]
        nearest_neighbours = []

        # Iterate through the other student to identify the k-nearest neighbours
        for other_student in other_students:
            distance = distance_function(
                student['semester_grades'], other_student['semester_grades'])

            insert_nearest_neighbours(nearest_neighbours, k, {
                'distance': distance,
                'student': other_student
            })

        votes = {final_grade: 0 for final_grade in range(1, 8)}

        # Determine the frequency of each category (ie. final grade)
        # among the k-nearest neighbours
        for nn in nearest_neighbours:
            votes[nn['student']['final_grade']] += 1

        prediction, most_votes = None, None

        # Determine which category received the most votes among the neighbours
        for final_grade in votes:
            if most_votes is None or most_votes < votes[final_grade]:
                prediction = final_grade
                most_votes = votes[final_grade]

        # Determine if the student was categorized correctly
        if prediction == student['final_grade']:
            num_correct += 1

        # Update the sum of squared errors
        sum_squared_errors += math.pow(prediction - student['final_grade'], 2)

    # Calculate the results of the model
    num_students = len(students)
    percent_correct = num_correct / num_students * 100
    hamming_loss = (num_students - percent_correct)/ num_students
    mean_squared_error = sum_squared_errors / num_students

    # Print the results
    print("percent_correct=", "%.2f" % percent_correct, ", mse=",
          "%.2f" % mean_squared_error, sep="")

    # Return the results
    return {
        "percent_correct": percent_correct,
        "hamming_loss": hamming_loss,
        "mean_squared_error": mean_squared_error
    }


def main():
    # Retrieve the raw data from the csv file
    raw_data = load_dataset('model_input_data.csv', print_file=False)
    raw_data_without_headers = raw_data[1:]

    # Organize the data into a list of dicts which each represent a student
    students = [{
        'id': student[0],
        'year': student[1],
        'semester_grades': [float(grade) for grade in student[2:-5]],
        'predicted_grade': int(student[-2]),
        'final_grade': int(student[-1])
    } for student in raw_data_without_headers]

    k_list, percent_correct_list, mean_squared_error_list = [], [], []

    for k in range(1, 31):
        print("for k=", k, ": ", sep="", end=" " if k < 10 else ""),

        results = k_nearest_neighbours_from_scratch(students, k)
        k_list.append(k)
        percent_correct_list.append(results["percent_correct"])
        mean_squared_error_list.append(results["mean_squared_error"])

    # plt.subplot(nrows, ncols, index, **kwargs)

    plt.figure(figsize=(7, 8))
    plt.subplots_adjust(hspace=1)

    plt.subplot(2, 1, 1)
    plt.title('Figure 1.1. Percent Correct vs. k-value')
    plt.xlabel('k')
    plt.ylabel('Percent Correct')
    plt.scatter(k_list, percent_correct_list)

    plt.subplot(2, 1, 2)
    plt.title('Figure 1.2. Mean Squared Error vs. k-value')
    plt.xlabel('k')
    plt.ylabel('Mean Squared Error')
    plt.scatter(k_list, mean_squared_error_list)

    plt.show()


##################################################
# Error is minimized when k=19
# for k=19: percent_correct=59.14, mse=0.60
##################################################

# for k=1:  percent_correct=46.59, mse=0.93
# for k=2:  percent_correct=45.16, mse=0.95
# for k=3:  percent_correct=50.90, mse=0.96
# for k=4:  percent_correct=52.33, mse=0.84
# for k=5:  percent_correct=51.61, mse=0.79
# for k=6:  percent_correct=52.33, mse=0.69
# for k=7:  percent_correct=55.91, mse=0.63
# for k=8:  percent_correct=53.76, mse=0.64
# for k=9:  percent_correct=55.56, mse=0.65
# for k=10: percent_correct=53.41, mse=0.65
# for k=11: percent_correct=54.12, mse=0.64
# for k=12: percent_correct=55.20, mse=0.63
# for k=13: percent_correct=54.48, mse=0.64
# for k=14: percent_correct=55.20, mse=0.63
# for k=15: percent_correct=56.27, mse=0.63
# for k=16: percent_correct=56.63, mse=0.62
# for k=17: percent_correct=57.35, mse=0.62
# for k=18: percent_correct=55.91, mse=0.63
# for k=19: percent_correct=59.14, mse=0.60
# for k=20: percent_correct=55.91, mse=0.63
# for k=21: percent_correct=55.91, mse=0.63
# for k=22: percent_correct=54.48, mse=0.65
# for k=23: percent_correct=56.63, mse=0.63
# for k=24: percent_correct=54.84, mse=0.66
# for k=25: percent_correct=56.63, mse=0.64
# for k=26: percent_correct=56.27, mse=0.63
# for k=27: percent_correct=56.63, mse=0.64
# for k=28: percent_correct=55.56, mse=0.65
# for k=29: percent_correct=56.27, mse=0.63
# for k=30: percent_correct=56.27, mse=0.63




if __name__ == '__main__':
    main()
