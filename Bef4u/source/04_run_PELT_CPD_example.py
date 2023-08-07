import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

# PELT: Pruned Exact Linear Time refers to a change point detection algorithm used in time series analysis.
# The algorithm detects locations where significant shifts are detected in the underlying statistical
# properties of the data. The changes can be shifts in mean, variance, distribution or other characteristics.
# PELT aims for computational efficiency in its search.


# Generate synthetic data with a single change point
n_samples = 1000
x = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(2, 1, 500)])
signal = np.reshape(x, (n_samples, 1))

# Perform change point detection using PELT
model = "l2"  # use l2-norm
algo = rpt.Pelt(model=model, min_size=10, jump=5).fit(signal)
result = algo.predict(pen=10)

# Plot the results
plt.plot(signal)
for i in result:
    plt.axvline(i, color="r", linestyle="--")
plt.title("PELT Change Point Detection")
plt.show()
a=1
