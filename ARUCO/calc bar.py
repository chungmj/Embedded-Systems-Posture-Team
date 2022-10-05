import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('out.csv', names=('x', 'y', 'z'))

x_value = df['x'].to_numpy()
y_value = df['y'].to_numpy()
threshold = 8

reps = 0
for i in range(len(y_value)):
    if y_value[i] > threshold and y_value[i - 1] < threshold:
        reps += 1
print(reps)

plt.plot(x_value, y_value)
plt.show()
