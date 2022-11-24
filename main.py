import math
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

    # Objective Function
    objective_function = []

    while True:
        # Plot the data and the averages
        plot_data(d)
        for average in averages:
            plt.plot(average[0], average[1], color='black', marker='o', linestyle='none')

        # Get a data point for the Objective Function
        objective_function.append(get_objective_function(d, averages))

        # Show the graph
        plt.title('k-means clustering for ' + str(k) + ' Clusters')
        # TODO: Uncomment
        # plt.show()

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
            # Plot Objective Function
            plt.plot(range(len(objective_function)), objective_function, 'ko')
            plt.plot(range(len(objective_function)), objective_function, 'k')
            plt.xlabel('Iteration')
            plt.ylabel('Sum of Error Squared')
            plt.title('Objective Function for ' + str(k) + ' Clusters')
            plt.show()

            # Return the centroids of the clusters
            return [averages, objective_function]


def get_objective_function(d, means):
    '''
    1) Get a distance as a radius
    2) Plot objective function on y-axis with iterations on x-axis
    '''

    sse = 0.0

    for r in d:
        # Error at point squared = ||xn - uk||^2
        sse += math.pow(get_distance([float(r[2]), float(r[3])], get_closest_mean([float(r[2]), float(r[3])], means)), 2)

    return sse


def get_decision_bounds(point1, point2):
    '''
    If y - a = m(x - b) will give the line containing both points 1 and 2, then y = `b - (x - a)/m` will be the line
    containing the midpoint between a and b and be perpendicular to the line containing a and b

    :param point1:
    :param point2:
    :return:
    '''
    # Intercept point (halfway between the two points in the parameters)
    x_constant = (point1[0] + point2[0]) / 2.0
    y_constant = (point1[1] + point2[1]) / 2.0

    # line
    x_axis = np.linspace(0.0, 7.0, 200)
    # Slope of the line (if y = mx + b for the line )
    slope = abs(point2[0] - point1[0]) / abs(point2[1] - point1[1])
    return y_constant - (x_axis - x_constant) / slope


def get_likelihood(point, cluster, clusters):

    distance_test = 1.0 / get_distance(point, cluster)
    distance_all = 0.0
    for c in clusters:
        distance_all += 1.0 / get_distance(point, c)

    return distance_test / distance_all


def plot_decision_boundaries(num_clusters):
    output = k_means_cluster(num_clusters, data)
    uk.append(output[0])
    d.append(output[1])
    plot_data(data)

    for index in range(len(uk[0])):
        p1 = [uk[0][index][0], uk[0][index][1]]
        plt.plot(p1[0], p1[1], 'ko', linestyle='none')

        for index2 in range((index + 1), len(uk[0])):
            p2 = uk[0][index2][0], uk[0][index2][1]

            x = (p1[0] + p2[0]) / 2.0
            y = (p1[1] + p2[1]) / 2.0

            # point
            # plt.plot(x, y, 'co', linestyle='none', linewidth=0.5)

            # line
            t = np.linspace(0.0, 7.0, 200)
            m = abs(p2[0] - p1[0]) / abs(p2[1] - p1[1])
            line = y - (t - x) / m

            # TODO: Use likelihood function to determine if it should graph or not
            print('(' + str(x) + ', ' + str(y) + ')')
            plt.plot(x, y, 'co', linestyle='none')
            l1 = get_likelihood([x, y], p1, uk[0])
            l2 = get_likelihood([x, y], p2, uk[0])
            l3 = 1.0 - l1 - l2
            print('L1:\t' + str(l1))
            print('L2:\t' + str(l2))
            print('L3:\t' + str(l3))

            if l3 < l1:
                plt.plot(t, line, 'c--')

    plt.show()

    pass


def mse(d, means):

    return get_objective_function(d, means) / 150.0

'''
3a)
average of the square of the distances between the centroid and the points over the entire population

Objective function divided by n
'''


def get_closest_mean(r, means):
    # returns the mean that is closest to the point
    distance = 99.9
    closest_mean = [0.0, 0.0]

    for mean in means:
        # Get distance between the current point and the mean
        d = get_distance([float(r[0]), float(r[1])], mean)

        # If current average is closer to the point than the prior average, make the prior average the current average
        if d < distance:
            distance = d
            closest_mean = [float(mean[0]), float(mean[1])]

    return closest_mean


def get_distance(pa, pb):
    # returns the distance between 2 points
    # return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
    return math.sqrt(math.pow(pa[0] - pb[0], 2) + math.pow(pa[1] - pb[1], 2))


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

    # k-means clustering
    k = [3]

    # uk = k_means_cluster(2, data)
    uk = []
    d = []

    plot_decision_boundaries(2)
    plot_decision_boundaries(3)

    '''
        output = k_means_cluster(num_clusters, data)
        uk.append(output[0])
        d.append(output[1])
        plot_data(data)

        for index in range(len(uk[0])):
            p1 = [uk[0][index][0], uk[0][index][1]]
            plt.plot(p1[0], p1[1], 'ko', linestyle='none')

            for index2 in range((index+1), len(uk[0])):
                p2 = uk[0][index2][0], uk[0][index2][1]

                x = (p1[0] + p2[0]) / 2.0
                y = (p1[1] + p2[1]) / 2.0

                # point
                # plt.plot(x, y, 'co', linestyle='none', linewidth=0.5)

                # line
                t = np.linspace(0.0, 7.0, 200)
                m = abs(p2[0] - p1[0]) / abs(p2[1] - p1[1])
                line = y-(t-x)/m

                # TODO: Use likelihood function to determine if it should graph or not
                print('(' + str(x) + ', ' + str(y) + ')')
                plt.plot(x, y, 'co', linestyle='none')
                l1 = get_likelihood([x, y], p1, uk[0])
                l2 = get_likelihood([x, y], p2, uk[0])
                l3 = 1.0 - l1 - l2
                print('L1:\t' + str(l1))
                print('L2:\t' + str(l2))
                print('L3:\t' + str(l3))

                if l3 < l1:
                    plt.plot(t, line, 'c--')

        plt.show()
        '''
    
    '''
    Plot decision boundaries for dataset using 
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
