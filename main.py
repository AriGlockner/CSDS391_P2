import numpy as np
import matplotlib.pyplot as plt
import csv

with open('CSDS391_P2\irisdata.csv', mode = 'r') as file:
    iris = csv.reader(file)
    x = 0

    for lines in iris:
        color = 'black'
        if lines[4] == 'setosa':
            color = 'blue'
        elif lines[4] == 'versicolor':
            color = 'red'
        elif lines[4] == 'virginica':
            color = 'green'
        else:
            color = 'black'

        plt.plot(lines, linestyle='none', marker='o', markerfacecolor=color)


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
