import pandas as pd
import numpy as np


df = pd.read_csv('out.csv', names=('x', 'y', 'z'))

y_value = df['y'].to_numpy()
threshold = 8


reps = 0
for i in range(len(y_value)):
    if y_value[i] > threshold and y_value[i-1] < threshold:
        count += 1
print(reps)


