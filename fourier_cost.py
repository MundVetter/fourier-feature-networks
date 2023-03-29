import numpy as np
def tune_progression(m, max, r = 1, epsilon = 0.01):
    if m == 2:
        return np.sqrt(max)
    r_now = (max/r)**(1/(m-1))
    if np.abs(r_now - r) < epsilon:
        return r_now
    else:
        return tune_progression(m, max, r_now)
    
def calculate_m_cost(max, m_min = 3, m_max = 10):
    r = []
    for m in range(m_min, m_max):
        r.append(tune_progression(m, max))
    return np.array(r) * np.arange(m_min, m_max)

_max = 1000 * 5
print(calculate_m_cost(_max, 2, 13), np.argmin(calculate_m_cost(_max, 2, 13)) + 2)
print(tune_progression(9, 1000* 5))