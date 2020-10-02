import numpy as np
import matplotlib.pyplot as plt
from step_func_draw import step_function

def relu(x):
    return np.maximum(x, 0)

x = np.arange(-5.0, 5.0, 0.1)
y1 = relu(x)
y2 = step_function(x)

plt.plot(x, y2, label = "ReLu")
plt.plot(x, y1, label = "step", lineStyle = '--')
plt.legend()
plt.ylim(-0.5, 5.0)
plt.show()