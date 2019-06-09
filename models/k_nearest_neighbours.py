import csv
import random
import math
import numpy as np
import matplotlib.pylab as plt

def load_dataset(filename, print_file=False):
    # Use a with/as block, which abstracts away opening and closing a file
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lines = list(csv_reader)
   
        if print_file:
            print('len(lines) =', len(lines))

            for line in lines:
                print(line)

        return lines


def euclidean_distance(point_1, point_2):
    sum_of_squares = 0

    for (val_1, val_2) in zip(point_1, point_2):
        sum_of_squares += math.pow(val_1 - val_2, 2)

    return math.sqrt(sum_of_squares)


def swap(a, b):
    a, b = b, a


def insert_nearest_neighbours(nearest_neighbours, k, new_neighbour):
    if len(nearest_neighbours) < k:
        nearest_neighbours.append(new_neighbour)
        return

    for i in range(k):
        if new_neighbour['distance'] < nearest_neighbours[i]['distance']:
            nearest_neighbours[i], new_neighbour = new_neighbour, nearest_neighbours[i]


def k_nearest_neighbours_model(students, k):
    num_correct = 0
    sum_squared_errors = 0

    for student in students:
        other_students = [s for s in students if s != student]
        nearest_neighbours = []

        for other_student in other_students:
            distance = euclidean_distance(
                student['semester_grades'], other_student['semester_grades'])

            insert_nearest_neighbours(nearest_neighbours, k, {
                'distance': distance,
                'student': other_student
            })

        votes = {final_grade:0 for final_grade in range(1, 8)}

        for nn in nearest_neighbours:
            votes[nn['student']['final_grade']] += 1

        # print(nearest_neighbours)

        model_prediction, most_votes = None, None

        for final_grade in votes:
            if most_votes is None or most_votes < votes[final_grade]:
                model_prediction = final_grade
                most_votes = votes[final_grade]

        # print('model_predicted=', most_common_key, ' actual=', student['final_grade'], ' id=', nearest_neighbours[0]['student']['id'])
        if model_prediction == student['final_grade']:
            num_correct += 1

        sum_squared_errors += math.pow(model_prediction-student['final_grade'], 2)

    num_students = len(students)
    percent_correct = num_correct / num_students * 100
    percent_error = 100 - percent_correct
    mean_squared_error = sum_squared_errors / num_students

    print("percent_correct=", "%.2f" % percent_correct, ", mse=", "%.2f" % mean_squared_error, sep="")
    return {
        "percent_correct": percent_correct,
        "percent_error": percent_error,
        "mean_squared_error": mean_squared_error
    }


def main():
    raw_data = load_dataset('model_input_data.csv', print_file=False)
    # print(raw_data)
    raw_data_without_headers = raw_data[1:]

    students = [{
        'id': student[0],
        'year': student[1],
        'semester_grades': [float(grade) for grade in student[2:-2]],
        'predicted_grade': int(student[-2]),
        'final_grade': int(student[-1])
    } for student in raw_data_without_headers]

    k_list, percent_correct_list, mean_squared_error_list = [], [], []

    for k in range(1, 31):
        print("for k=", k, ": ", sep="", end=" " if k < 10 else ""),

        results = k_nearest_neighbours_model(students, k)
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
