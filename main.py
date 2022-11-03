import numpy as np
import matplotlib.pyplot as plt

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
