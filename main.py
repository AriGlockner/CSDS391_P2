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
        #plt.plot(float(r[1]), float(r[2]), linestyle='none', marker='o', color=c)
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

    means = []
    points = []
    num_data_points = len(d)

    # Initialize the first means of the data set
    for i in range(k):
        means.append([float(d[int(num_data_points * i / k)][1]), float(d[int(num_data_points * i / k)][2])])
        points.append([])

    # Repeat updating the mean
    for a in range(5):
        # Categorize each data point to its closest mean and update that mean's coordinates
        for r in range(num_data_points):
            # get the data point
            pt = [float(d[k][1]), float(d[r][2])]

            # find the mean closest to it
            distance = get_distance(pt, means[0])
            mean = 0

            # check for closer distances
            for i in range(k):
                temp_distance = get_distance(pt, means[i])
                if temp_distance < distance:
                    mean = i
                    distance = temp_distance

            # categorize the point to a mean
            points[mean].append(pt)

        # Update mean by shifting it to the average for each item in the cluster
        for i in range(k):
            average_x = 0.0
            average_y = 0.0

            for j in points[i]:
                average_x += j[0]
                average_y += j[1]
            if len(points[i]) != 0:
                means[i] = [average_x / len(points[i]), average_y / len(points[i])]

    return means


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
            x_distance += abs(float(r[1]) - mean[0])
            y_distance += abs(float(r[2]) - mean[1])

    return [x_distance / num_points, y_distance / num_points]


def get_closest_mean(r, means):
    # returns the mean that is closest to the point
    distance = 99.9
    closest_mean = [0.0, 0.0]

    for mean in means:
        # Get distance between the current point and the mean
        d = get_distance([float(r[1]), float(r[2])], mean)

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
    plot_data(data)

    '''
    # k-means clustering
    k = [1, 2, 3, 4, 5]
    t = np.linspace(0, 2 * pi, 200)

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
            D = get_objective_function(data, point, uk)
            plt.plot(point[0] + D[0] * np.cos(t), point[1] + D[1] * np.sin(t), '-', color='gray')
        
        Plot decision boundaries
        Possible ideas:
        1. Take averages between points and use that to get lines (vernoi diagram)
        2. Use likelihood function
        3. k neighbors/centroids
        Go back and reread the lecture slides for another idea
        '''

    # plt.show()

    # Show the plot
    plt.show()
