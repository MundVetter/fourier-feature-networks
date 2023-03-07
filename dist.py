import numpy as np
import matplotlib.pyplot as plt

# v = np.random.normal(size=(1000))
# v2 = v * 10

# def count_small(arr, lim):
#     return np.sum(np.abs(arr) < lim)

# lim = 0.1
# print(count_small(v, lim))
# print(count_small(v2, lim))
# bins = np.linspace(-25, 25, 5000)
# plt.hist(v, bins=bins, alpha=0.5, label='v')
# plt.hist(v2, bins=bins, alpha=0.5, label='v2')
# plt.legend(loc='upper right')
# # limit view to -1 and 1
# # plt.xlim(-1, 1)
# plt.show()

def input_encoder(x, b):
    return np.cos((2.*np.pi*x[...,None]) * b)

# high_freq = np.linspace(1000, 2000, 5000)
g = 4
N = 512
# high_freq = 1/((1/(N//2))*g**np.arange(0, 7))
# high_freq = 1/((1/(N//2))*g**np.arange(0, 7))
high_freq = 1/(1/1000*g**np.arange(0, 20))

# high_freq = np.abs(np.random.normal(0, 1, size=(128)))
# embed_dim = 128
# omega = np.arange(embed_dim // 2, dtype=float)
# omega /= embed_dim / 2.
# high_freq = 1/10_000**omega

x = np.arange(0, N)
# get all high freq of above 20
res = input_encoder(x, high_freq[high_freq > 0.3])
# res = input_encoder(x, high_freq)

# res = input_encoder(x, high_freq[:len(high_freq)//2])
# res = input_encoder(x, high_freq)

# for each row subtract the other rows and check if all are zero
# is so print the row number
for i in range(res.shape[0]):
    diff =res[i] - res[i+1:]
    # check all rows in diff with all close zero
    for j, row in enumerate(diff):
        if np.allclose(row, 0, rtol=0, atol=0.05):
            print(f'Found simmilar rows at {i} and {j}')

# calculate dot product simmilarty for a row
# sim = [[np.dot(row, r) / np.linalg.norm(row) for r in res] for row in res]
# plt.imshow(sim)
# plt.show()