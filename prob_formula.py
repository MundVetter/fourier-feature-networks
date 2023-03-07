import random
import matplotlib.pyplot as plt
from math import erf, sqrt
from scipy.stats import norm
from scipy.special import comb

def simulate_normal_distribution(x, z, mu, sigma):
    values = []
    for i in range(z):
        values.append(sigma * random.gauss(mu, 1.0))
    count = sum(1 for value in values if value > y or value < -y)
    return count >= x

def prob_x_above_or_below_y(x, y, n, sigma):
    p = norm.sf(y/sigma) + norm.cdf(-y/sigma)
    prob = comb(n, x) * (p ** x) * ((1 - p) ** (n - x))
    return prob

# Set the values for testing
x = 1
n = 100
y = 250
sigma = 115

# Calculate the theoretical probability
prob = prob_x_above_or_below_y(x, y, sigma, n)

# Simulate the probability
simulations = 100_000
successes = sum(1 for i in range(simulations) if simulate_normal_distribution(x, n, 0, sigma))

# Print the results
print(f"Theoretical probability: {prob:.5f}")
print(f"Simulated probability: {successes/simulations:.5f}")

# Plot the distribution of simulated values
x = 0