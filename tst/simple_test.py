import numpy as np

weights = []
weights.append(np.array([[1, 2, 3, 4], [.1, .2, .3, .4]]))
weights.append(np.array([[5, 6, 7, 8], [.5, .6, .7, .8]]))

print(weights)

s = np.sum(weights, axis=0)

print(s)
