import numpy as np
import matplotlib.pyplot as plt
from numpy import double

'''
x = range(100)

# 1 random sample
y = np.random.random_sample(100)
for i in range(100):
    y[i] += i/5
plt.plot(x, y)

# Average of 10 random samples:
y = []

for j in range(1):
    y.append((np.random.random_sample(100) + np.random.random_sample(100) + np.random.random_sample(100) +
        np.random.random_sample(100) + np.random.random_sample(100) + np.random.random_sample(100) +
        np.random.random_sample(100) + np.random.random_sample(100) + np.random.random_sample(100) + np.random.random_sample(100))/10)
    
for i in range(100):
    y[0][i] += i/5

plt.plot(x, y[0])

plt.show()
'''

'''
rng = 10

# likelihood (h1, h2, h3, h4, h5)
h12345 = [0, 0.25, 0.5, 0.75, 1.0]

# Prior (P(d|hi))
hi = [[0.1], [0.2], [0.4], [0.2], [0.1]]

# P(hi|d) and P(X|d)
xD = []
y = [[], [], [], [], []]

for j in range(rng):
    for i in range(5):
        v = hi[i][j] * h12345[i]
        hi[i].append(v)
    sum = hi[0][j] + hi[1][j] + hi[2][j] + hi[3][j] + hi[4][j]
    xD.append(sum)

    if sum > 0:
        for i in range(5):
            y[i].append(hi[i][j] / sum)
    else:
        for i in range(5):
            y[i].append(0)

# Plot P(DN+1 = lime|d1,…,dN):
plt.plot(range(rng), y[0], 'red')
plt.plot(range(rng), y[1], 'green')
plt.plot(range(rng), y[2], 'blue')
plt.plot(range(rng), y[3], 'gray')
plt.plot(range(rng), y[4], 'purple')

plt.xlabel('Number of observations in d')
plt.ylabel('probability of hypothesis')
plt.yticks(np.arange(0.0, 1.01, 0.25))
plt.show()

# Plot P(hi|d1,…,dN):
plt.subplot(121)
plt.plot(range(rng + 1), hi[0], 'red')
plt.plot(range(rng + 1), hi[1], 'green')
plt.plot(range(rng + 1), hi[2], 'blue')
plt.plot(range(rng + 1), hi[3], 'gray')
plt.plot(range(rng + 1), hi[4], 'purple')

plt.yticks(np.arange(0.0, 0.41, 0.1))
plt.show()
'''

# likelihood
h12345 = [0, 0.25, 0.5, 0.75, 1.0]

yAvg = [[], [], [], [], []]

iterations = 100

for iteration in range(iterations):
    x = [0, 0, 0, 0, 0]
    y = [[], [], [], [], []]
    n = 0

    for j in range(100):
        n = n+1

        for i in range(5):
            rv = np.random.uniform(0, 1, 1)

            if rv <= h12345[i]:
                x[i] = x[i] + 1
            y[i].append((x[i] + 1) / (n + 2))

    if len(yAvg[0]) == 0:
        yAvg = y
    else:
        for i in range(5):
            for j in range(100):
                yAvg[i][j] += y[i][j]

for i in range(5):
    for j in range(100):
        yAvg[i][j] /= iterations

for i in range(5):
    plt.plot(range(100), yAvg[i])

plt.show()
