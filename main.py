import random

import numpy as np
import matplotlib.pyplot as plt
import csv

from numpy import pi


def plot_data(d):
    for r in d:
        # Get the color based on the species
        c = 'black'
        if r[4] == 'setosa':
            c = 'blue'
        elif r[4] == 'versicolor':
            c = 'red'
        elif r[4] == 'virginica':
            c = 'green'
        else:
            c = 'black'

        # plot the species
        plt.plot(float(r[2]), float(r[3]), linestyle='none', marker='o', color=c)

    # Format Graph
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    pass


def k_means_cluster(k, d):
    '''
    uki = Sum(n, rnk * xni) / Sum(n, rnk)
    xn = nth data vector
    rnk is 1 if xn is in the kth class, otherwise 0

    Algorithm:
    1) Initialize k points as means
    2) Categorize each data point to its closest mean and update the mean's coordinates
    (which are the averages of the number of items categorized in that cluster so far)
    3) repeat the process for a given number of iterations and at the end we have our clusters
    '''

    # Plot the data
    plot_data(d)

    # Initialize k different random points to be the initial means
    averages = k * [[0.0, 0.0]]
    for index in range(k):
        # Makes sure that two initial means cannot be the same starting point
        while True:
            random_point = random.randint(0, 150)
            if not averages.__contains__([float(d[random_point][2]), float(d[random_point][3])]):
                break
        # Set the average cluster in the averages array and plot the point
        averages[index] = [float(d[random_point][2]), float(d[random_point][3])]
        plt.plot(averages[index][0], averages[index][1], linestyle='none', marker='o', color='black')

    plt.show()

    while True:
        # Make sure that each iteration is changing the averages
        last_averages = averages.copy()

        # Categorize each data point to its closest mean
        points = k * [[]]

        for r in d:
            point = [float(r[2]), float(r[3])]
            # Get the average point that is closest to the current point
            closest_average = get_closest_mean(point, averages)
            print(str(point) + ' ' + str(closest_average) + ' ' + str(averages))

            print(averages.index(closest_average))
            points[averages.index(closest_average)].append(point)

        # Update each mean's coordinates
        for index in range(k):
            x_average = 0.0
            y_average = 0.0
            print('Length: ' + str(len(points[index])) + '\tIndex: ' + str(index))
            for point in points[index]:
                x_average += point[0]
                y_average += point[1]

            averages[index] = [x_average/float(len(points[index])), y_average/float(len(points[index]))]
            print(averages[index])

        # Plot the data
        plot_data(d)

        # Plot the averages
        for average in averages:
            plt.plot(average[0], average[1], linestyle='none', marker='o', color='black')

        plt.show()

        if last_averages == averages:
            return averages

    '''
    means = []
    points = []

    # Initialize the first means of the data set
    for i in range(k):
        means.append([float(d[int(i * len(d) / k)][2]), float(d[int(i * len(d) / k)][3])])
        points.append([])

    count = 0

    # Repeat updating the mean
    while True:
        print(count)
        count += 1
        last_means = means.copy()

        # Empty points
        points = []
        for i in range(k):
            points.append([])

        # Categorize each data point to its closest mean and update that mean's coordinates
        for r in range(len(d)):
            # get the data point
            pt = [float(d[k][2]), float(d[r][3])]

            ''
            # find the mean closest to it
            distance = get_distance(pt, means[0])
            mean = 0

            # check for closer distances
            for i in range(k):
                temp_distance = get_distance(pt, means[i])
                if temp_distance < distance:
                    mean = i
                    distance = temp_distance
            ''
            mean = get_closest_mean(pt, means)
            index = -1

            for i in range(len(means)):
                if mean == means[i]:
                    index = i
                    break

            # categorize the point to a mean
            points[index].append(pt)

        # Update mean by shifting it to the average for each item in the cluster
        for i in range(k):
            average_x = 0.0
            average_y = 0.0

            for j in points[i]:
                average_x += j[0]
                average_y += j[1]
            if len(points[i]) != 0:
                means[i] = [average_x / len(points[i]), average_y / len(points[i])]

        if means == last_means:
            return means
    '''

    return averages

def get_objective_function(d, mean, means):
    # Returns the objective function
    x_distance = 0.0
    y_distance = 0.0
    num_points = 0.0

    for r in d:
        # Get the closest average point
        closest_mean = get_closest_mean(r, means)

        # If the closest average point is the average point
        if np.array_equal(mean, closest_mean):
            num_points += 1.0
            x_distance += pow(abs(float(r[2]) - mean[0]), 2)
            y_distance += pow(abs(float(r[3]) - mean[1]), 2)

    return [x_distance / num_points, y_distance / num_points]


def get_closest_mean(r, means):
    # returns the mean that is closest to the point
    distance = 99.9
    closest_mean = [0.0, 0.0]

    for mean in means:
        # Get distance between the current point and the mean
        # TODO: might need to change from [0, 1] back to [2, 3]
        d = get_distance([float(r[0]), float(r[1])], mean)

        # If current average is closer to the point than the prior average, make the prior average the current average
        if d < distance:
            distance = d
            closest_mean = [float(mean[0]), float(mean[1])]

    return closest_mean


def get_distance(pa, pb):
    # returns the distance between 2 points
    return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])


with open('CSDS391_P2\irisdata.csv') as file:
    # Used to take out the header from the file
    heading = next(file)

    # iris data
    iris = csv.reader(file)
    data = []

    # Get the iris data stored as variable data
    for row in iris:
        # Add the iris data to the data 2D array
        data.append(row)

    # plot data
    # plot_data(data)

    # k-means clustering
    # k = [1, 2, 3, 4]
    # t = np.linspace(0, 2 * pi, 200)

    uk = k_means_cluster(2, data)

    # for num_clusters in k:
        # uk = k_means_cluster(num_clusters, data)

    '''
    for numMeans in k:
        # Plot data points
        plot_data(data)

        # Get averages
        uk = k_means_cluster(numMeans, data)

        # For each mean, plot the mean and the objective function
        for point in uk:
            # Plot averages:
            plt.plot(point[0], point[1], linestyle='none', marker='o', color='black')

            # Plot Objective Function:
            # = get_objective_function(data, point, uk)
            #plt.plot(point[0] + D[0] * np.cos(t), point[1] + D[1] * np.sin(t), '-', color='gray')

        plt.show()
    '''

    '''
    Plot decision boundaries
    Possible ideas:
    1. Take averages between points and use that to get lines (vernoi diagram)
    2. Use likelihood function
    3. k neighbors/centroids
    Go back and reread the lecture slides for another idea
    '''

    '''
    Exercise 2B:
    just do an equation to approximate linear bound
    '''

    # Show the plot
    # plt.show()
