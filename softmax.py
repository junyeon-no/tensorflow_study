import numpy as np
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    exp_sum = np.sum(exp_a)
    y = exp_a/ exp_sum
    return y

if __name__ == "__main__":
    a = np.array([0.3, 2.9, 4.0])
    print( softmax(a) )
    
    a2 = np.array([1010, 1000, 990])
    print(softmax(a2))
