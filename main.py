import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

'''
Clustering Methods
'''


def plot_data(d, is_title=False, title=''):
    for r in d:
        # Get the color based on the species

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
    if is_title:
        plt.title(title)
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


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def plot_neural_network_decision_boundary(d, m, b, point):
    # timescale
    timescale = np.linspace(3.0, 7.0, 200)

    # Plot
    plot_data(d)
    plt.plot(timescale, m * timescale + b, 'c')
    plt.ylim(0.9, 2.6)
    plt.title('Decision Boundary for the Non-Linearity Above Overlaid on the Iris Data')
    plt.show()

    # z = mx - y + b
    return 1.0 - sigmoid(m * point[0] - point[1] + b)


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
        plt.plot(x, y, sigmoid(m * x - y + b), 'o', color=color)

    plt.show()

    pass


def show_simple_classifier(d, m, b, k):
    v = d[k]
    print('Point: ' + str(v))
    print(sigmoid(m * float(v[2]) - float(v[3]) + b))

    pass


'''
Neural Networks Methods
'''


def plot_data_and_line(d, w0, w1, w2, color='c', marker='solid', show=False):
    plot_data(d)
    plt.axline(xy1=(0, -w0 / w2), xy2=(-w0 / w1, 0), color=color, linestyle=marker)

    plt.xlim(2.9, 7.1)
    plt.ylim(0.9, 2.6)

    if show:
        plt.show()

    pass


def get_point_actual(flower):
    if flower == 'versicolor':
        return 0.0
    else:
        return 1.0


def mse(d, w0, w1, w2, color='c', marker='-', plot=True):
    if not plot:
        plot_data_and_line(d, w0, w1, w2, color=color, marker=marker)
    # z = w0 + w1 * x1 + w2 * x2
    # E = 1/2 * sum of n for (yn(xn, W1:L) - tn)^2
    E = 0.0
    for r in d:
        x1 = float(r[2])
        x2 = float(r[3])

        z = w0 + w1 * x1 + w2 * x2
        sigma = sigmoid(z)

        v = get_point_actual(r[4])

        # print(v, sigma)

        E += math.pow(v - sigma, 2)

    return E / (2.0 * len(d))


def compute_mse_for_2_points(d, w00, w10, w20, w01, w11, w21):
    # plot_data_and_line(d, w00, w10, w20, 'c', 'solid')
    mse0 = mse(d, w00, w10, w20, 'c', 'solid')
    # plt.show()
    # plot_data_and_line(d, w01, w11, w21, 'c', 'solid')
    mse1 = mse(d, w01, w11, w21, 'r', 'dashed')
    plt.show()
    return 0 # [mse0, mse1]


def compute_summed_gradient(d, w0, w1, w2, plot=True):
    '''
    f(z) = 1/(1 + e-z)
    f'(z) = (e-z)/(1 + e-z)2
    z = mx - y + b

    f'(x) = m(e(y - mx - b))/(1 + e(y - mx - b))2
    f'(y) = -(e(y - mx - b))/(1 + e(y - mx - b))2
    '''

    if plot:
        plot_data_and_line(d, w0, w1, w2)

    grad_w0 = 0.0
    grad_w1 = 0.0
    grad_w2 = 0.0

    for r in d:
        # Get the point
        x1 = float(r[2])
        x2 = float(r[3])

        z = w0 + w1 * x1 + w2 * x2

        v = get_point_actual(r[4])

        # Calculate Sigma
        sigma = sigmoid(z)
        # print(v, sigma)

        # Calculate the derivatives of the sigmas
        d_sigma = math.exp(-z)/math.pow((1 + math.exp(-z)), 2)

        # Calculate the gradient of z
        df_dz = 2.0 * (v - sigma) * -d_sigma

        # Sum up the gradient at each point
        grad_w0 += df_dz * 1
        grad_w1 += df_dz * x1
        grad_w2 += df_dz * x2

    # Divide by the number of data points
    grad_w0 /= len(d)
    grad_w1 /= len(d)
    grad_w2 /= len(d)
    # print([grad_w0, grad_w1, grad_w2])

    #
    step_size = -10.0
    if plot:
        plot_data_and_line(d, w0 + grad_w0 * step_size, w1 + grad_w1 * step_size, w2 + grad_w2 * step_size, 'r')
        plt.show()

    return [grad_w0, grad_w1, grad_w2]


'''
Learning a Decision Boundary Through Optimization
'''


def optimize_gradient(d, w0, w1, w2, plot_learning_curve=False):
    step = 0.1
    stopping_criteria = 0.01
    learning_curve = []

    while True:
        # Calculate the Gradient with respect to z
        g = compute_summed_gradient(d, w0, w1, w2, False)
        # Calculate the Gradient with respect to w0, w1, and w2 respectively
        w0 -= g[0] * step
        w1 -= g[1] * step
        w2 -= g[2] * step

        # Calculate the norm and store the norm in the learning curve list
        norm = math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        learning_curve.append(norm)

        # Stop optimizing the gradient when the norm is less than the stopping criteria
        if norm < stopping_criteria:
            if plot_learning_curve:
                return [[w0, w1, w2], learning_curve]
            return [w0, w1, w2]


def show_optimize_gradient(d, w0, w1, w2, show_curve=True):
    # Plot the Initial line
    plot_data_and_line(d, w0, w1, w2)

    # Plot the gradient curve
    if show_curve:
        # Plot the final line
        line, curve = optimize_gradient(d, w0, w1, w2, True)
        plot_data_and_line(d, line[0], line[1], line[2], 'r')
        plt.show()

        # Plot the gradient curve
        plt.plot(range(len(curve)), curve, 'k')
        plt.xlabel('Iterations')
        plt.ylabel('Norm of Gradient')
        plt.show()
        return line, curve

    # Don't Plot the Gradient Curve
    else:
        # Plot the final line
        line = optimize_gradient(d, w0, w1, w2)
        plot_data_and_line(d, line[0], line[1], line[2], 'r')
        plt.show()
        return line


def random_show_optimize_gradient(d):
    # Return the output of the show_optimize_gradient method using a random seed
    np.random.seed(1234)
    w = np.random.uniform(-5, 5, 3)
    return show_optimize_gradient(d, w[0], w[1], w[2], True)


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

    # Exercises: 1a, 1b, 1c, and 1d for k = 2
    plot_decision_boundaries(2, data, t)
    # Exercises: 1a, 1b, 1c, and 1d for k = 3
    plot_decision_boundaries(3, data, t)

    '''
    Linear Decision Boundaries
    '''

    # Get data for just the 2nd and 3rd iris classes
    v_data = []
    start = 50
    while start < 150:
        v_data.append(data[start])
        start += 1

    # Exercise 2a
    plot_data(v_data, True, 'Versicolor and Virginica Iris Data')
    plt.show()
    # Exercise 2b
    print(sigmoid(-0.5 * 4.7 - 1.1 + 4.1))
    # Exercise 2c
    print(plot_neural_network_decision_boundary(v_data, -0.6, 4.8, [5.5, 2.0]))
    # Exercise 2d
    plot_neural_network(v_data, -0.6, 4.8)

    # Exercise 2e
    print()
    show_simple_classifier(v_data, -0.6, 4.8, 84)
    show_simple_classifier(v_data, -0.6, 4.8, 99)
    show_simple_classifier(v_data, -0.6, 4.8, 10)
    show_simple_classifier(v_data, -0.6, 4.8, 67)
    print()

    '''
    Neural Networks
    '''

    # Exercise 3a
    print('mse:')
    print(mse(v_data, -45, 6, 10))
    print(mse(v_data, -44, 7, 11))
    print()

    # Exercise 3b
    # print(compute_mse_for_2_points(v_data, -45, 6, 10, -30, 1, 15))
    plot_data_and_line(v_data, -45, 6, 10, 'c', 'solid', True)
    plot_data_and_line(v_data, -30, 1, 15, 'c', 'solid', True)
    
    # Exercise 3e
    compute_summed_gradient(v_data, -45, 6, 10)
    compute_summed_gradient(v_data, -44, 7, 11)

    '''
    Learning a Decision Boundary Through Optimization
    '''

    # Exercise 4a
    print(optimize_gradient(v_data, -44, 7, 11))
    # Exercise 4b
    show_optimize_gradient(v_data, -44, 7, 11)
    # Exercise 4c
    random_show_optimize_gradient(v_data)
