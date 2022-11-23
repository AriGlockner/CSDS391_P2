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
    # Initialize k different random points to be the initial means
    averages = k * [[0.0, 0.0]]
    for index in range(k):
        # Makes sure that two initial means cannot be the same starting point
        while True:
            random_point = random.randint(0, 150)
            if not averages.__contains__([float(d[random_point][2]), float(d[random_point][3])]):
                break
        # Set the average cluster in the averages array
        averages[index] = [float(d[random_point][2]), float(d[random_point][3])]

    while True:
        # Plot the data and the averages
        plot_data(d)
        for average in averages:
            plt.plot(average[0], average[1], color='black', marker='o', linestyle='none')
        plt.show()

        # Make sure that each iteration is changing the averages
        last_averages = averages.copy()

        # Categorize each data point to its closest mean
        averages = [[0.0, 0.0] for i in range(k)]
        num_points = k * [0.0]
        for r in d:
            # Classify what average the point is closest to
            index = last_averages.index(get_closest_mean([r[2], r[3]], last_averages))

            # Increase how many points are assigned to a specific average
            num_points[index] += 1.0

            # Add the x and y positions to the average
            averages[index][0] += float(r[2])
            averages[index][1] += float(r[3])

        # Make the sums an average by dividing by the number of points that are closest to a specific average
        for index in range(k):
            averages[index][0] /= num_points[index]
            averages[index][1] /= num_points[index]

        # If the cluster points did not shift locations, return the clusters
        if last_averages == averages:
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
    k = [1, 2, 3, 4]
    # t = np.linspace(0, 2 * pi, 200)

    # uk = k_means_cluster(2, data)

    for num_clusters in k:
        uk = k_means_cluster(num_clusters, data)

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
