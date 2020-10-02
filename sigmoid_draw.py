import numpy as np
import matplotlib.pyplot as plt
from step_func_draw import step_function

def sigmoid(x):
    return 1/(1 + np.exp(-x))

if __name__ == "__main__":
    x = np.arange(-5.0, 5.0 , 0.1)
    y = sigmoid(x)
    plt.plot(x, y, label = "sigmoid")
    plt.ylim(-0.1, 1.1)

    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y, linestyle = "--", label = "setp")
    plt.legend()
    plt.show()