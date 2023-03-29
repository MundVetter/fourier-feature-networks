import numpy as np

# def compute_overlapping_grid_2d(n, dim, o):
#     s = o / n
#     areas = n ** dim
#     # create a array starting at [0, .., 0] and ending at [n, .., n] with length dim
#     arr = np.arange(n)
#     # create a meshgrid with the array
#     mesh = np.meshgrid(*[arr] * dim)
#     # create a list of all the points in the meshgrid
#     points = np.vstack(map(np.ravel, mesh)).T
#     return np.stack(points, points + s)

def compute_bounds(n, o):
    s = o / n
    r = np.arange(n) / n
    return np.stack((r, r + s), axis=1)

def compute_value_region(x, region_boundaries):
    """ Compute the value of the embedding"""
    x = np.clip(x - region_boundaries[:, 0], a_min=0, a_max=region_boundaries[:, 1] - region_boundaries[:, 0])
    x = x / (region_boundaries[:, 1] - region_boundaries[:, 0])
    return x % 1

def compute_value(data, bounds):
    x = compute_value_region(data[:, 0], bounds)
    y = compute_value_region(data[:, 1], bounds)
    x2 = np.concatenate([x, x > 0])
    # repeat x2 for len(y) times
    x2 = np.repeat(x2, len(y))

    y2 = np.concatenate([y, y > 0])
    # repeat y2 for len(x) times
    y2 = np.repeat(y2, len(x))

    return x2 * y2


x = np.array([[0.1, 0.5]])
bounds = compute_bounds(10, 1.1)
z = compute_value(x, bounds)
print(z)

# import matplotlib.pyplot as plt
# # random color per 4 points
# colors = np.random.rand(len(squares) // 4, 3)
# colors = np.repeat(colors, 4, axis=0)
# plt.scatter(squares[:, 0], squares[:, 1], c=colors, s=1000, alpha=0.5)
# plt.show()