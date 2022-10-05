import pandas as pd
import numpy as np

df = pd.read_csv('out.csv', names=('x', 'y', 'z'))

y_value = df['y'].to_numpy()
print(y_value)
max_y = np.max(y_value)
min_y = np.min(y_value)

print(max_y)
print(min_y)