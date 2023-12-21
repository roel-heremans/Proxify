# Databricks notebook source
# MAGIC %pip install surpyval

# COMMAND ----------

from scipy.stats import weibull_min
import surpyval
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample from Weibull

# COMMAND ----------

# sample from Weibull
n = 100     # number of samples
k = 2.4     # shape
lam = 5.5   # scale

x = weibull_min.rvs(k, loc=0, scale=lam, size=n)

# Reproduce params without censoring
model_no_censoring = surpyval.Weibull.fit(x=x)
print('Shape parameter estimate: {:.2f} versus generated {}'.format(model_no_censoring.beta, k))
print('Scale parameter estimate: {:.2f} versus generated {}'.format(model_no_censoring.alpha, lam))

# Collect the xcn for the Right-censored data with a cutoff after 8 "months"
cutoff = 8
x_in = [[i, i+1] for i in np.arange(0, cutoff)]
x_in.append(cutoff)  # adding cutoff time

c_in = [2 for i in np.arange(0, cutoff)]
c_in.append(1) # adding right censor indicator

[bincount, binedges] = np.histogram(x[x<cutoff], bins=np.arange(0, cutoff+1), density=False)
n_in = np.append(bincount, len(x[x>cutoff]))

print('Checking of the inputs\n*****************************')
print('Service call times, \t x: {}'.format(x_in))
print('Censor inputs, \t\t c: {}'.format(c_in))
print('Service call count, \t n: {}'.format(n_in))

# fit right-censored data
model_with_censoring = surpyval.Weibull.fit(x=x_in, c=c_in, n=n_in)
print('Shape parameter estimate (No censoring, With censoring, Simulated): ({:.2f}, {:.2f}, {})'.format(model_no_censoring.beta, model_with_censoring.beta,k))
print('Scale parameter estimate (No censoring, With censoring, Simulated): ({:.2f}, {:.2f}, {})'.format(model_no_censoring.alpha,model_with_censoring.alpha, lam))

#showing the result
print('Synthetic data')
print('Generated sample size: {}'.format(n))
print('Generated Shape value: {}'.format(k))
print('Generated Scale value: {}'.format(lam))

plt.stairs(bincount/sum(bincount), binedges)
# Plot the estimated Weibull probability density function
xn = np.linspace(0, 12, 100)
pdf = weibull_min.pdf(xn, model_with_censoring.beta, scale=model_with_censoring.alpha)
plt.plot(xn, pdf, 'r', label='Est PDF from Censored:\n (shape, scale)=({:.2f},{:.2f})'.format(model_with_censoring.beta,model_with_censoring.alpha))
pdf_orig = weibull_min.pdf(xn, k, scale=lam)
plt.plot(xn, pdf_orig, 'g', label='PDF Original')
plt.xlabel('Service call time')
plt.ylabel('Probability Density')
plt.title('Weibull Distribution')
plt.legend()

plt.show()