import math
import random

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

from numpy import pi

'''
Clustering Methods
'''


def plot_data(d):
    for r in d:
        # Get the color based on the species
        c = 'black'
        if r[4] == 'setosa':
            c = 'red'
        elif r[4] == 'versicolor':
            c = 'green'
        elif r[4] == 'virginica':
            c = 'blue'
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
            random_point = random.randint(0, len(d))
            if not averages.__contains__([float(d[random_point][2]), float(d[random_point][3])]):
                break
        # Set the average cluster in the averages array
        averages[index] = [float(d[random_point][2]), float(d[random_point][3])]

    # Objective Function
    objective_function = []

    # Number of iterations
    iterations = 0

    while True:
        iterations += 1

        if len(d) == 150:
            # Plot the data and the averages
            plot_data(d)
            for average in averages:
                plt.plot(average[0], average[1], color='black', marker='o', linestyle='none')

            # Get a data point for the Objective Function
            objective_function.append(get_objective_function(d, averages))

            # Show the graph
            plt.title('k-means clustering for ' + str(k) + ' Clusters\nIteration: ' + str(iterations))
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
            if len(d) == 150:
                # Plot Objective Function
                plt.plot(range(len(objective_function)), objective_function, 'ko')
                plt.plot(range(len(objective_function)), objective_function, 'k')
                plt.xlabel('Iteration')
                plt.ylabel('Sum of Error Squared')
                plt.title('Objective Function for ' + str(k) + ' Clusters')
                plt.show()

                # Return the centroids of the clusters
                return [averages, objective_function]
            else:
                return [averages, 0]


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


def get_decision_bounds(point1, point2, t):
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

    # Slope of the line (if y = mx + b for the line )
    slope = abs(point2[0] - point1[0]) / abs(point2[1] - point1[1])
    return y_constant - (t - x_constant) / slope


def get_likelihood(point, cluster, clusters):
    # distance_test is 1 / the distance from the point to the cluster
    distance_test = 1.0 / get_distance(point, cluster)

    # distance_all is 1 / the sum of the distance from the point to each cluster
    distance_all = 0.0
    for c in clusters:
        distance_all += 1.0 / get_distance(point, c)

    # return the percentage likelihood of it being a specific cluster at a specific point
    return distance_test / distance_all


def plot_decision_boundaries(num_clusters, iris_data, t):
    uk = []
    d = []

    # Get clusters
    output = k_means_cluster(num_clusters, iris_data)
    uk.append(output[0])
    d.append(output[1])
    plot_data(iris_data)

    for index in range(len(uk[0])):
        p1 = [uk[0][index][0], uk[0][index][1]]
        plt.plot(p1[0], p1[1], 'ko', linestyle='none')

        for index2 in range((index + 1), len(uk[0])):
            p2 = uk[0][index2][0], uk[0][index2][1]

            x = (p1[0] + p2[0]) / 2.0
            y = (p1[1] + p2[1]) / 2.0

            # line
            m = abs(p2[0] - p1[0]) / abs(p2[1] - p1[1])
            line = y - (t - x) / m

            # Determine if the line should be plotted
            l1 = get_likelihood([x, y], p1, uk[0])
            l3 = 1.0 - (2.0 * l1)
            if l3 < l1:
                plt.plot(t, line, 'c:')

    # Make plot fancier and show the plot
    names = ['Setosa', 'Versicolor', 'virginica', 'Cluster', 'Decision Boundaries']
    colors = ['r', 'g', 'b', 'k', 'c']
    hands = []

    for i in range(5):
        hands.append(mpatches.Patch(color=colors[i], label=names[i]))

    plt.legend(handles=hands, loc='upper left')

    if len(iris_data) == 150:
        plt.xlim(0.0, 7.1)
        plt.ylim(0.0, 2.6)
    else:
        plt.xlim(2.9, 7.1)
        plt.ylim(0.9, 2.6)

    plt.title('Decision Boundaries for ' + str(num_clusters) + ' Clusters')
    plt.show()

    pass


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


'''
Linear Decision Boundaries Methods
'''


def signoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def compute_linear_classification(point, c0, c1, d):
    """
    :param c1: cluster 1
    :param c0: cluster 2
    :param d: dataset
    :param point: the point to get calculate the percent likelihood of being verginica
    :return: a number between 0 and 1
    """

    t = np.linspace(3.0, 7.0, 200)

    boundary = get_decision_bounds(c0, c1, t)

    # Plot
    plot_data(d)
    plt.plot(c0[0], c0[1], 'ko')
    plt.plot(c1[0], c1[1], 'ko')
    plt.plot(t, boundary, 'c')
    plt.ylim(0.9, 2.6)

    # Slope of the boundary line
    m = -1/(abs(c0[0] - c1[0]) / abs(c0[1] - c1[1]))

    # b0 from boundary line
    x0 = c0[0] + c1[0]
    y0 = c0[1] + c1[1]
    b0 = y0 + x0/m

    # b1 = y + x/m -> from point
    b1 = point[1] + point[0]/m

    #
    # y = mx + b0 -> boundary
    # y = b1 - x/m
    # b1 - x/m = mx + b0
    # mx + x/m = (b1 - b0)
    # x(m + 1/m) = (b1 - b0)
    x = (b1 - b0) / (m + 1/m)
    y = b1 - (b1 - b0) / (m * m + 1)
    print(x)
    print(y)
    # plt.plot(x, y, 'yo')
    # plt.plot(point[0], point[1], 'ro')
    # plt.plot(-y, -x, 'yo')
    plt.plot(point[0], point[1], 'yo')

    # boundary = y_constant - (t - x_constant) / slope
    print(signoid(0))

    plt.show()

    '''
    w = get_distance(c0, c1) / 2.0
    
    # Get the distance to each cluster
    d0 = get_distance(point, c0)
    d1 = get_distance(point, c1)

    ''
    x0 = point[0] - c0[0]
    x1 = point[0] - c1[0]
    x = x0 - x1

    y0 = point[1] - c0[1]
    y1 = point[1] - c1[1]
    y = y0 - y1
    ''

    # Calculate the classification function
    if d0 < d1:
        z = (d0 - d1) / d0
        # x /= x0
        # y /= y0
    else:
        z = (d0 - d1) / d1
        # x /= x1
        # y /= y1
    # z = math.sqrt(x * x + y * y)
    return 1.0 / (1.0 + math.exp(-z))
    '''
    return 0.0


def plot_neural_network_decision_boundary(d, m, b):
    # timescale
    timescale = np.linspace(3.0, 7.0, 200)

    # Plot
    plot_data(d)
    plt.plot(timescale, m * timescale + b, 'c')
    plt.ylim(0.9, 2.6)
    plt.show()

    '''
    w = get_distance(c0, c1) / 2.0

    # Get the distance to each cluster
    d0 = get_distance(point, c0)
    d1 = get_distance(point, c1)

    ''
    x0 = point[0] - c0[0]
    x1 = point[0] - c1[0]
    x = x0 - x1

    y0 = point[1] - c0[1]
    y1 = point[1] - c1[1]
    y = y0 - y1
    ''

    # Calculate the classification function
    if d0 < d1:
        z = (d0 - d1) / d0
        # x /= x0
        # y /= y0
    else:
        z = (d0 - d1) / d1
        # x /= x1
        # y /= y1
    # z = math.sqrt(x * x + y * y)
    return 1.0 / (1.0 + math.exp(-z))
    '''
    return 0.0


def plot_neural_network(d, m, b):
    ax = plt.axes(projection='3d')
    for v in d:
        x = float(v[2])
        y = float(v[3])
        # choose a color
        if v[4] == 'versicolor':
            color = 'b'
        else:
            color = 'g'

        # plot
        plt.plot(x, y, signoid(m * x - y + b), 'o', color=color)

    plt.show()

    pass


def show_simple_classifier(d, m, b, k):
    v = d[k]
    print('Point: ' + str(v))
    print(signoid(m * float(v[2]) - float(v[3]) + b))

    pass


'''
Neural Networks Methods
'''


def mse(d, m, b):
    # E = 1/2 * sum of n for (yn(xn, W1:L) - tn)^2
    E = 0.0
    for r in d:
        if r[4] == 'versicolor':
            E += math.pow(1 - signoid(m * float(r[2]) - float(r[3]) + b), 2)
        else:
            E += math.pow(signoid(m * float(r[2]) - float(r[3]) + b), 2)

    return E / 2.0


def exc3b(d, m1, b1, m2, b2):
    plot_data(d)

    t = np.linspace(3.0, 7.0, 200)

    plt.plot(t, m1 * t + b1, 'c-')
    plt.plot(t, m2 * t + b2, 'y-')
    print(mse(d, m1, b1))
    print(mse(d, m2, b2))

    pass


def exc3c():

    pass


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
    t = np.linspace(0.0, 7.0, 200)

    '''
    Clustering
    '''
    # TODO: Uncomment
    '''
    # Exercises: 1a, 1b, 1c, and 1d for k = 2
    plot_decision_boundaries(2, data, t)
    # Exercises: 1a, 1b, 1c, and 1d for k = 3
    plot_decision_boundaries(3, data, t)
    '''
    
    '''
    Linear Decision Boundaries
    '''
    # Get data for just the 2nd and 3rd iris classes
    v_data = []
    start = 50
    while start < 150:
        v_data.append(data[start])
        start += 1

    # TODO: Uncomment
    '''
    # Exercise 2a
    plot_data(v_data)
    plt.show()

    # Exercise 2b
    # print(signoid(-0.5 * 4.7 - 1.1 + 4.1))

    # Exercise 2c
    plot_neural_network_decision_boundary(v_data, -0.6, 4.8)

    # Exercise 2d
    plot_neural_network(v_data, -0.6, 4.8)
    
    # Exercise 2e
    show_simple_classifier(v_data, -0.6, 4.8, 84)
    show_simple_classifier(v_data, -0.6, 4.8, 99)
    show_simple_classifier(v_data, -0.6, 4.8, 10)
    show_simple_classifier(v_data, -0.6, 4.8, 67)
    '''

    '''
    Neural Networks
    '''

    # print(mse(v_data, -0.6, 4.8))

    # ex2c(v_data, -0.5, 4.1)
    # Exercise 3a
    # print(mse(v_data, -0.5, 4.1))

    # Exercise 3b
    # exc3b(v_data, -0.6, 4.8, -0.3, 3.3)

    plt.show()

