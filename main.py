import numpy as np
import matplotlib.pyplot as plt
import csv

with open('CSDS391_P2\irisdata.csv') as file:
    # Used to take out the header from the file
    heading = next(file)

    # iris data
    iris = csv.reader(file)

    # Deriving a learning rule -> uk = Sum of n (r * x) / Sum of n (r)
    ukx = [0.0, 0.0, 0.0]
    uky = [0.0, 0.0, 0.0]
    rnk = [0.0, 0.0, 0.0]

    # Getting
    D = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Plot the species
    for row in iris:
        # Get the color based on the species
        color = 'black'
        if row[4] == 'setosa':
            color = 'blue'
            ukx[0] += float(row[1])
            uky[0] += float(row[2])
            rnk[0] += 1
        elif row[4] == 'versicolor':
            color = 'red'
            ukx[1] += float(row[1])
            uky[1] += float(row[2])
            rnk[1] += 1
        elif row[4] == 'virginica':
            color = 'green'
            ukx[2] += float(row[1])
            uky[2] += float(row[2])
            rnk[2] += 1
        else:
            color = 'black'
        # plot the species
        plt.plot(float(row[1]), float(row[2]), linestyle='none', marker='o', markerfacecolor=color)

    for i in range(3):
        ukx[i] /= rnk[i]
        uky[i] /= rnk[i]

    plt.plot(ukx, uky, linestyle='none', marker='o', markerfacecolor='black')

    # Objective function
    
    # petal_length,petal_width
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.show()


'''
# Get Random x and y values
minValue = 0
maxValue = 1000
size = 100

x = np.random.randint(minValue, maxValue + 1, size)
y = np.random.randint(minValue, maxValue + 1, size)

# plot x and y
plt.plot(x, y, linestyle='none', marker='o', markerfacecolor='blue', markersize=3)

# Make Graph Fancier
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot")


# Show the graph
plt.show()
'''
