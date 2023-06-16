import numpy as np

it = 22
a = np.load(str(it) + '.npy')

a = np.unique(a)
kd = np.exp(a * (-1) / (0.00198720425864083 * 298.15)).flatten()

ind = np.argsort(kd)
ind = ind[:200]

b = kd[ind]
print(np.mean(b), np.std(b))
print(b)
