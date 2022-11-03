import numpy as np
import matplotlib.pyplot as plt

# Get Random x and y values
minValue = 0
maxValue = 1000
size = 50
x = np.random.random_integers(minValue, maxValue, size)
y = np.random.random_integers(minValue, maxValue, size)

# plot x and y
plt.plot(x, y, linestyle='none', marker='o', markerfacecolor='blue', markersize=5)

# Make Graph Fancier
plt.xlabel("x")
plt.ylabel("y")

plt.show()