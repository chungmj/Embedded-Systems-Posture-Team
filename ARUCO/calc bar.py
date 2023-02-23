import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

df = pd.read_csv('out.csv', names=('x', 'y', 'z'))

x_value = df['x'].to_numpy()
y_value = df['y'].to_numpy()
z_value = df['z'].to_numpy()
threshold = 8

reps = 0
for i in range(len(y_value)):
    if y_value[i] > threshold and y_value[i - 1] < threshold:
        reps += 1
print(reps)

plt.plot(x_value, y_value)
plt.xlim(-20, 20)
plt.ylim(-20, 40)
plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# # ax.plot3D(x_value, y_value, z_value, 'gray')
# ax.scatter3D(x_value, y_value, z_value, c=z_value, cmap='Greens')
#
# plt.show()