import numpy as np
import matplotlib.pyplot as plt

s = np.random.poisson(4, 2000*2)

count, bins, ignored = plt.hist(s, 100, density=True)
plt.show()
