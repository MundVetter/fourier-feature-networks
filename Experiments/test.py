import numpy as np
import matplotlib.pyplot as plt

# Define parameters for grid creation
s = 0.1  # size of grid areas
n = 50  # minimum number of points in each area
o = 5  # minimum number of extra points on each side of an area

# Generate some example data for visualization
x = np.random.normal(loc=0.5, scale=0.3, size=1000)
y = np.random.normal(loc=-0.5, scale=0.2, size=1000)
data = np.vstack((x, y)).T

# Create grid areas
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

x_edges = np.arange(x_min, x_max + s, s)
y_edges = np.arange(y_min, y_max + s, s)

for i in range(1, len(x_edges) - 1):
    for j in range(1, len(y_edges) - 1):
        x1, x2 = x_edges[i - 1], x_edges[i]
        y1, y2 = y_edges[j - 1], y_edges[j]
        
        area = data[(x1 <= data[:, 0]) & (data[:, 0] <= x2) &
                    (y1 <= data[:, 1]) & (data[:, 1] <= y2)]
        
        if len(area) < n:
            # merge with neighbouring areas until minimum number of points is reached
            while len(area) < n and i > 1:
                i -= 1
                x1 = x_edges[i - 1]
                area = data[(x1 <= data[:, 0]) & (data[:, 0] <= x2) &
                            (y1 <= data[:, 1]) & (data[:, 1] <= y2)]
            while len(area) < n and i < len(x_edges) - 2:
                i += 1
                x2 = x_edges[i]
                area = data[(x1 <= data[:, 0]) & (data[:, 0] <= x2) &
                            (y1 <= data[:, 1]) & (data[:, 1] <= y2)]
                
            # increase area size until minimum number of extra points on each side is reached
            while (x1 > x_min and np.sum((x1 - s <= data[:, 0]) & (data[:, 0] <= x1)) < o) or \
                  (x2 < x_max and np.sum((x2 <= data[:, 0]) & (data[:, 0] <= x2 + s)) < o) or \
                  (y1 > y_min and np.sum((y1 - s <= data[:, 1]) & (data[:, 1] <= y1)) < o) or \
                  (y2 < y_max and np.sum((y2 <= data[:, 1]) & (data[:, 1] <= y2 + s)) < o):
                if x1 > x_min and np.sum((x1 - s <= data[:, 0]) & (data[:, 0] <= x1)) < o:
                    x1 -= s
                if x2 < x_max and np.sum((x2 <= data[:, 0]) & (data[:, 0] <= x2 + s)) < o:
                    x2 += s
                if y1 > y_min and np.sum((y1 - s <= data[:, 1]) & (data[:, 1] <= y1)) < o:
                     y1 -= s
                if y2 < y_max and np.sum((y2 <= data[:, 1]) & (data[:, 1] <= y2 + s)) < o:
                    y2 += s
                area = data[(x1 <= data[:, 0]) & (data[:, 0] <= x2) &
                            (y1 <= data[:, 1]) & (data[:, 1] <= y2)]
    
        # plot grid area and data points
        plt.plot([x1, x2], [y1, y1], 'k-', lw=0.5)
        plt.plot([x1, x2], [y2, y2], 'k-', lw=0.5)
        plt.plot([x1, x1], [y1, y2], 'k-', lw=0.5)
        plt.plot([x2, x2], [y1, y2], 'k-', lw=0.5)
        plt.plot(area[:, 0], area[:, 1], 'bo', ms=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Overlapping grid visualization')
plt.show()