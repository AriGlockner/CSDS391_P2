import numpy as np
import matplotlib.pyplot as plt
import csv


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
        plt.plot(float(r[1]), float(r[2]), linestyle='none', marker='o', color=c)

    # Format Graph
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    pass

def k_means_cluster(k, data):
    '''
    uki = Sum(n, rnk * xni) / Sum(n, rnk)
    xn = nth data vector
    rnk is 1 if xn is in the kth class, otherwise 0

    Algorithm:
    1) Initialize k points (use 1st k points as means)
    2) Categorize each data point to its closest mean and update the mean's coordinates
    (which are the averages of the number of items categorized in that cluster so far)
    3) repeat the process for a given number of iterations and at the end we have our clusters
    '''

    means = []
    points = []
    num_data_points = len(data)

    # Initialize the first k-points to become the means of the data set
    for i in range(k):
        means.append([float(data[int(num_data_points * i / k)][1]), float(data[int(num_data_points * i / k)][2])])
        points.append([])

    # Repeat updating the mean
    for a in range(5):
        # Categorize each data point to its closest mean and update that mean's coordinates
        for r in range(num_data_points):
            # get the data point
            pt = [float(data[k][1]), float(data[r][2])]

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
        print(means)

    return means

# returns the distance between 2 points


def get_distance(pa, pb):
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

    # k-means clustering
    k = [1, 2, 3, 4, 5]

    for numMeans in k:
        plot_data(data)
        uk = k_means_cluster(numMeans, data)

        for point in uk:
            plt.plot(point[0], point[1], linestyle='none', marker='o', color='black')

        plt.show()

    # Show the plot
    plt.show()
